@echo off
rem Launch Chrome with the DevTools debugging port open so the transcriber can
rem see which tab is playing a video.
rem
rem Chrome 136+ deliberately refuses --remote-debugging-port when running on the
rem default user profile, so this uses a dedicated profile directory. The first
rem time you use it you will need to sign in to your course site inside this
rem window. Your normal Chrome is untouched.

rem Machine-wide installs land in Program Files; an install without admin rights
rem goes under the user's own AppData instead.
set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" (
  echo Could not find chrome.exe. Edit launch_chrome.cmd and set CHROME manually.
  pause
  exit /b 1
)

set "PROFILE=%LOCALAPPDATA%\lectureTranscriber\chrome-profile"

rem This port must match chrome.debug_port in config.toml, which is what the
rem daemon connects to. 9222 is the default on both sides.
start "" "%CHROME%" ^
  --remote-debugging-port=9222 ^
  --remote-debugging-address=127.0.0.1 ^
  --user-data-dir="%PROFILE%" ^
  --no-first-run ^
  --no-default-browser-check %*
