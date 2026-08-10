[Setup]
AppName=collagenSim
AppVersion=0.1.0
AppPublisher=collSIm project
DefaultDirName={autopf}\collagenSim
DefaultGroupName=collagenSim
OutputDir=..\dist
OutputBaseFilename=collagenSim-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\collagenSim.exe

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\collagenSim\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\collagenSim"; Filename: "{app}\collagenSim.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall collagenSim"; Filename: "{uninstallexe}"
Name: "{autodesktop}\collagenSim"; Filename: "{app}\collagenSim.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\collagenSim.exe"; Description: "Launch collagenSim"; Flags: postinstall nowait skipifsilent
