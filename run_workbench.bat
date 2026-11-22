@echo off
REM Launch the Tkinter diagnostic workbench on local Windows.
REM Requires Python 3.11+ with Tcl/Tk selected in the installer.
python -m pip install --user numpy
python -m k8ssim.gui
