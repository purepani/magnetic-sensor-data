{
  description = "Moment Reconstruction of Image";

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

    python = let
        packageOverrides = self: super: {
          scipy = super.scipy.overridePythonAttrs(old: rec {
            version = "1.8.0";
            src =  super.fetchPypi {
              pname = "scipy";
              inherit version;
              sha256 = "MdTy1rckvJqY5Se1hJuKflib8epjDDOqVj7akSyf8L0=";
             };
          });
        };
             in pkgs.python3.override {inherit packageOverrides; self = python;};

        packageName = "MagneticHeadTracking";
      in {
#        packages.${packageName} = app;

#        defaultPackage = self.packages.${system}.${packageName};
#        python.withPackages (ps: [ps.numpy ps.scipy ps.pandas])
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ 
              (python.withPackages (ps: [ps.numpy ps.scipy ps.pandas ps.matplotlib ps.seaborn]))
              python39Packages.scikit-learn
              python39Packages.einops
              python39Packages.dill
              #python3Packages.numpy
              #python3Packages.scipy
              #python3Packages.pandas
              #python39Packages.jedi-language-server
              pkgs.nodePackages.pyright
              pkgs.mypy
          ];
#          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}

