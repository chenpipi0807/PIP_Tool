@echo off
echo 启动Excel转JSON工具...
python "%~dp0xlsx2json.py"
if errorlevel 1 (
    echo 发生错误，按任意键退出...
    pause > nul
)
