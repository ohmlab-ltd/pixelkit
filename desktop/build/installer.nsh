; The first-run runtime is outside the install dir, so the stock uninstaller leaves it.
!macro customUnInstall
  ${ifNot} ${isUpdated}
    RMDir /r "$LOCALAPPDATA\PixelKit\runtime"
  ${endIf}
!macroend
