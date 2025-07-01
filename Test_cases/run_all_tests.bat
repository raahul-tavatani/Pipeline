@echo off
echo === Running All Test Cases ===

echo ==== Activate Python 3.8 ENV ====
REM Activate the Python 3.8 virtual environment
call C:\Pipeline\py_env_carla38\Scripts\activate.bat

REM Now you're inside the Python 3.8 virtual environment and should see (py_env_carla38) in the prompt

REM Confirm which Python is being used
python --version

echo ==== Running Test Controller ====
::C:\Pipeline\py_env_carla38\Scripts\python.exe "C:\Pipeline\Test_cases\run_all_tests.py"

::echo ==== Converting all PCD files to BIN ====
::C:\Pipeline\py_env_carla38\Scripts\python.exe "C:\Pipeline\Scripts\convert_pcd_to_kitti_bin.py"

echo ==== Running Batch Prediction Pipeline ====
C:\Pipeline\py_env_carla38\Scripts\python.exe "C:\Pipeline\Scripts\batch_predict.py" --cfg_file "C:\Pipeline\Scripts\cfgs\kitti_models\pv_rcnn.yaml" --ckpt "C:\Pipeline\models\pv_rcnn_8369.pth"

echo === All steps completed ===
pause
