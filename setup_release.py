"""Prepare the pinned hover release. This command never opens a radio."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts
            or '\\' in value or ':' in value or '\x00' in value):
        raise ValueError('Release paths must be relative POSIX paths.')
    return path


def project_path(value):
    path = ROOT.joinpath(*relative_path(value).parts)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Release path leaves the project.')
    return path


def verify_tree(root, build_id, allow_config=False):
    report = json.loads((root / 'portable_release_manifest.json').read_text())
    if f"{report['build_id']:08x}" != build_id:
        raise ValueError('Installed release build ID differs.')
    for name, expected in report['file_sha256'].items():
        path = root.joinpath(*relative_path(name).parts)
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('Release file is missing or leaves the release: ' + name)
        if allow_config and name == 'flight_config.json':
            continue
        if digest(path) != expected:
            raise ValueError('Release file hash differs: ' + name)
    return report


def extract_verified(archive, staging, build_id):
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        seen = set()
        for member in members:
            rel = relative_path(member.filename)
            if (rel.parts[0] != 'RL_takeoff' or member.filename in seen
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe or duplicate archive member.')
            if len(rel.parts) < 2 and not member.is_dir():
                raise ValueError('Archive file is outside RL_takeoff.')
            seen.add(member.filename)
        if sum(member.file_size for member in members) > 536870912:
            raise ValueError('Release archive exceeds the extraction limit.')
        package.extractall(staging)
    root = staging / 'RL_takeoff'
    report = verify_tree(root, build_id)
    expected = set(report['file_sha256']) | {'portable_release_manifest.json'}
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    if actual != expected:
        raise ValueError('Archive inventory differs from its release manifest.')
    return root


def prepare(archive_override=None):
    manifest = json.loads((ROOT / 'CURRENT_HOVER_RELEASE.json').read_text())
    target = project_path(manifest['active_project'])
    firmware = project_path(manifest['root_flash_archive'])
    archive = Path(archive_override).resolve() if archive_override else project_path(manifest['archive'])
    if target.exists():
        verify_tree(target, manifest['build_id'], allow_config=True)
    else:
        if not archive.is_file():
            if archive_override:
                raise FileNotFoundError('The specified release archive does not exist.')
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    print('Downloading the pinned hover release...', flush=True)
                    request = urllib.request.Request(manifest['download_url'],
                        headers={'User-Agent': 'RL_takeoff-release'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, stream)
                if digest(temporary) != manifest['archive_sha256']:
                    raise ValueError('Downloaded release archive hash differs.')
                os.replace(temporary, archive)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if digest(archive) != manifest['archive_sha256']:
            raise ValueError('Release archive hash differs.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.hover-install-', dir=target.parent) as directory:
            stage = extract_verified(archive, Path(directory), manifest['build_id'])
            if target.exists():
                raise FileExistsError('Another installation created the target; retry verification.')
            os.rename(stage, target)
    actual_firmware = target.joinpath(*relative_path(manifest['active_flash_archive']).parts)
    if digest(actual_firmware) != manifest['firmware_zip_sha256']:
        raise ValueError('Installed firmware hash differs.')
    if firmware.exists():
        if digest(firmware) != manifest['firmware_zip_sha256']:
            raise ValueError('Root firmware differs from the retained release.')
    else:
        firmware.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual_firmware, firmware)
    return {'prepared': True, 'build_id': manifest['build_id'],
            'release_directory': manifest['active_project'],
            'firmware_sha256': manifest['firmware_zip_sha256'],
            'radio_opened': False, 'flash_executed': False, 'flight_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an existing pinned release ZIP instead of downloading.')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.archive), indent=2))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print('Blocked: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
