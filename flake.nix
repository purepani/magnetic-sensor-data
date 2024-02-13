{
  description = "Moment Reconstruction of Image";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    devenv,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {};

      packageName = "MagneticHeadTracking";
    in {
      devShell = devenv.lib.mkShell {
        inherit inputs pkgs;

        modules = [
          ({pkgs, ...}: let
            python' = pkgs.python3.override {
              packageOverrides = self: super: {
                matplotlib = super.matplotlib.override {enableQt = true;};
              };
            };
            #matplotlib = pkgs.python3Packages.matplotlib.override {enableQt = true;};
          in {
            env.QT_PLUGIN_PATH = with pkgs.qt5; "${qtbase}/${qtbase.qtPluginPrefix}";
            packages = [
              pkgs.zlib
              (python'.withPackages
                (ps: [ps.matplotlib]))
              pkgs.nodejs
              pkgs.libgccjit
              pkgs.xorg.libX11
              pkgs.libGLU
              pkgs.libGL
              pkgs.ffmpeg
              pkgs.xorg.libXrender
            ];

            languages.python = {
              enable = true;
              venv.enable = true;
              poetry.enable = true;
            };
            languages.typescript = {
              enable = true;
            };
          })
        ];
      };
    });
}
