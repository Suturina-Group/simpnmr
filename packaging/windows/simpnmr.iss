; Inno Setup script for the SimpNMR Windows installer.
;
; Compile (on Windows, after PyInstaller has produced dist\SimpNMR\):
;     iscc /DAppVersion=2.0.0 packaging\windows\simpnmr.iss
;
; Produces packaging\windows\Output\SimpNMR-Setup-<version>.exe — a standard
; double-click installer that creates a Start-menu shortcut (with icon), an
; optional desktop shortcut, and an uninstaller.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "SimpNMR"
#define AppPublisher "Suturina Group"
#define AppURL "https://simpnmr.org"
#define AppExeName "SimpNMR.exe"

[Setup]
AppId={{6D8B7C2E-3F4A-4B1C-9E5D-2A7F1B0C4D9E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install by default so no admin rights are required.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=SimpNMR-Setup-{#AppVersion}
SetupIconFile=simpnmr.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The entire PyInstaller one-folder output. The build script places this at
; dist\SimpNMR relative to the repository root; the path below is relative to
; this .iss file (packaging\windows).
Source: "..\..\dist\SimpNMR\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
