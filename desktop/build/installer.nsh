; Slim-install cleanup: the first-run bootstrap puts the Python runtime
; (2-8 GB) under %LOCALAPPDATA%\PixelKit\runtime — outside both the
; install dir and userData, so the stock uninstaller would strand it.
; Guarded so an in-place UPDATE never wipes the runtime.
!macro customUnInstall
  ${ifNot} ${isUpdated}
    RMDir /r "$LOCALAPPDATA\PixelKit\runtime"
  ${endIf}
!macroend
