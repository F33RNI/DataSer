import os
import platform

import PyInstaller.config


PyInstaller.config.CONF['workpath'] = './build'
# -*- mode: python ; coding: utf-8 -*-


# Final name (without version and platform name)
COMPILE_NAME = 'DataSer'

# Version of main.py
COMPILE_VERSION = "1.1.0"

# Files and folders to include in final build directory (dist/COMPILE_NAME folder)
INCLUDE_FILES = ['gui.ui',
                 'README.md',
                 'Screenshot.png',
                 'icon.ico',
                 'icon.png',
                 'config.json',
                 'LICENSE']

_datas = []
for include_file in INCLUDE_FILES:
    if os.path.isdir(include_file):
        _datas.append((include_file, os.path.basename(include_file)))
    else:
        _datas.append((include_file, '.'))
print("datas: {}".format(str(_datas)))

_name = COMPILE_NAME + '-' + COMPILE_VERSION + '-' + str(platform.system() + '-' + str(platform.machine()))

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['_bootlocale'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=_name,
)
