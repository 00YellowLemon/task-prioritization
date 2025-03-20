{ pkgs, ... }: {
  channel = "stable-24.05";

  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.fastapi
    pkgs.python311Packages.uvicorn
    pkgs.python311Packages.pandas # Example, add your other dependencies
    pkgs.python311Packages.requests # Example, add your other dependencies
  ];

  env = {};

  idx = {
    previews = {
      enable = true;
      previews = {
        web = {
          command = [ "./devserver.sh" ];
          env = { PORT = "$PORT"; };
          manager = "web";
        };
      };
    };
    extensions = [ "ms-python.python" "rangav.vscode-thunder-client" ];
    workspace = {
      onCreate = {
        install = "python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt";
        default.openFiles = [ "app.py" ];
      };
    };
  };
}