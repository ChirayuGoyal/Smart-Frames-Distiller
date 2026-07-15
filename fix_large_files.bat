@echo off
setlocal
cd /d "%~dp0"
echo Working dir: %CD%

REM Ensure ignore rules exist (videos + models)
if not exist .gitignore (
  echo ERROR: .gitignore missing
  exit /b 1
)

echo.
echo === Fetching remote ===
git fetch origin
if errorlevel 1 (
  echo FETCH FAILED
  exit /b 1
)

echo.
echo === Rewriting local history from origin/main ^(keeps LICENSE^) ===
git reset --soft origin/main
if errorlevel 1 (
  echo Soft reset failed - is origin/main available?
  exit /b 1
)

REM Unstage everything, then re-add respecting .gitignore
git reset HEAD
git add -A

echo.
echo === Staged files ^(should NOT include .mp4 / .onnx / FREmbeddings^) ===
git status

echo.
echo Checking for forbidden large paths still staged...
git diff --cached --name-only | findstr /i /r "\.mp4$ \.onnx$ FREmbeddings output\.mp4 output2\.mp4"
if not errorlevel 1 (
  echo.
  echo ERROR: Large files are still staged. Fix .gitignore and re-run.
  exit /b 1
)

echo.
echo === Committing clean tree ===
git commit -m "Add Smart Frames Distiller implementation (exclude videos and model weights)"
if errorlevel 1 (
  echo Commit failed or nothing to commit
  exit /b 1
)

echo.
echo === Pushing to origin/main ===
git push -u origin main
if errorlevel 1 (
  echo PUSH FAILED
  exit /b 1
)

echo.
echo === DONE ===
git log --oneline -5
git status -sb
endlocal
