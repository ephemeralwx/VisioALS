; VisioALS Inno Setup Script
; Produces a single Setup.exe that installs VisioALS on the user's machine.

#define MyAppName      "VisioALS"
#define MyAppVersion   "1.0.0"
#define MyAppPublisher "VisioALS"
#define MyAppURL       "https://visioals.com"
#define MyAppExeName   "VisioALS.exe"

[Setup]
AppId={{B3F7A2D1-8C4E-4F6A-9D2B-1E5F8A3C7D9E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=VisioALS_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Uncomment the line below when you have an icon file:
; SetupIconFile=assets\visioals.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Include everything PyInstaller put in dist\VisioALS
Source: "dist\VisioALS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";         Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}";   Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  TrackingPage: TWizardPage;
  EyeRadio: TNewRadioButton;
  HeadRadio: TNewRadioButton;
  InfoLabel: TNewStaticText;

procedure InitializeWizard;
begin
  TrackingPage := CreateCustomPage(wpSelectTasks,
    'Tracking Mode', 'Choose how you want to control VisioALS.');

  InfoLabel := TNewStaticText.Create(TrackingPage);
  InfoLabel.Parent := TrackingPage.Surface;
  InfoLabel.Caption := 'You can change this later by pressing M in the app.';
  InfoLabel.Top := 0;
  InfoLabel.Left := 0;
  InfoLabel.Width := TrackingPage.SurfaceWidth;

  EyeRadio := TNewRadioButton.Create(TrackingPage);
  EyeRadio.Parent := TrackingPage.Surface;
  EyeRadio.Caption := 'Eye Tracking — move your eyes to control the cursor';
  EyeRadio.Top := InfoLabel.Top + InfoLabel.Height + 24;
  EyeRadio.Left := 0;
  EyeRadio.Width := TrackingPage.SurfaceWidth;
  EyeRadio.Checked := True;

  HeadRadio := TNewRadioButton.Create(TrackingPage);
  HeadRadio.Parent := TrackingPage.Surface;
  HeadRadio.Caption := 'Head Tracking — rotate your head to control the cursor';
  HeadRadio.Top := EyeRadio.Top + EyeRadio.Height + 12;
  HeadRadio.Left := 0;
  HeadRadio.Width := TrackingPage.SurfaceWidth;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigDir, ConfigPath, Mode, Content: String;
begin
  if CurStep = ssPostInstall then
  begin
    if HeadRadio.Checked then
      Mode := 'head'
    else
      Mode := 'eye';
    ConfigDir := ExpandConstant('{userappdata}\VisioALS');
    ForceDirectories(ConfigDir);
    ConfigPath := ConfigDir + '\config.json';
    Content := '{' + #13#10 +
      '  "tracking_mode": "' + Mode + '"' + #13#10 +
      '}';
    SaveStringToFile(ConfigPath, Content, False);
  end;
end;
