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
  } @inputs:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      customOverrides = self: super: {
        # Overrides go here
      };


      packageName = "MagneticHeadTracking";
    in {
      #        packages.${packageName} = app;

      #        defaultPackage = self.packages.${system}.${packageName};
      #        python.withPackages (ps: [ps.numpy ps.scipy ps.pandas])
      devShell = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          ({pkgs, ...}: {
            packages = [pkgs.zlib];
            languages.python = {
              enable = true;
              venv.enable = true;
              poetry.enable = true;
            };
            languages.typescript = {
              enable=true;
            };

          })
        ];
          #(python3.withPackages (ps: [ps.numpy ps.scipy ps.pandas ps.matplotlib ps.seaborn ps.einops ps.psutil ps.notebook ps.dill ps.scikit-learn]))];
      };
    });
}
