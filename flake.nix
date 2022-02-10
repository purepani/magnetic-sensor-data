{
  description = "Tracking the motion of a head with magnetic sensors";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        customOverrides = self: super: {
          # Overrides go here
        };


        packageName = "MagneticHeadMotionTracking";
      in {
#        packages.${packageName} = app;

#        defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ 
              python3
              python3Packages.scikitimage
              python3Packages.scipy
              python3Packages.sympy
              python3Packages.numpy
              #python39Packages.jedi-language-server
              pkgs.nodePackages.pyright
              pkgs.mypy
          ];
#          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}

