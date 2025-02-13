# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['HR_Estimator_Online_multiprocessing.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Add the shape predictor file
        ('shape_predictor_68_face_landmarks.dat', '.'),  

        # Add the checkpoint model file
        ('checkpoint_MMSE/MTTS_CSTM_MMSE_T_10_shift_0.625_combined_loss_best_model_1.pth', 
         'checkpoint_MMSE'),  

        # Add the IMG_Source folder
        ('IMG_Source', 'IMG_Source'),  
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.distribution'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HR_Estimator_Online_multiprocessing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HR_Estimator_Online_multiprocessing',
)
