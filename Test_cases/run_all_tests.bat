@echo off
echo === Running All Test Cases ===

REM Activate your virtual environment if needed
REM call C:\OpenPCDet-Win11-Compatible/win_env_openpcdet/Scripts\activate.bat

REM Run the Python test controller
python "C:\Pipeline\Test_cases\run_all_tests.py"

echo ==== Converting all PCD files to BIN ====
python "C:\Pipeline\Scripts\convert_pcd_to_kitti_bin.py"

echo === All tests completed ===
pause
