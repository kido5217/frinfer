{
  description = "FrInfer development environment (CUDA 13.1, Python 3.13)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # The CUDA toolkit is unfree (CUDA EULA).
      pkgs = (import nixpkgs) {
        inherit system;
        config.allowUnfree = true;
      };

      cuda = pkgs.cudaPackages_13_1;
      python = pkgs.python313.withPackages (
        ps: with ps; [
          jinja2
          jsonschema
          numpy
          pytest
          pyyaml
          rich
          safetensors
          torch
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "ninfer-dev";
        # Workstation-local HF-hub cache holding the converted .ninfer artifacts used by the real
        # Engine tests (see tests/README.md): $NINFER_ARTIFACT_HUB/models--<repo>/snapshots/<rev>/.
        env.NINFER_ARTIFACT_HUB = "/home/kido/trash/ai/models/hf/hub";
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.just
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_profiler_api
          cuda.cuda_nvtx
          cuda.nsight_compute
          cuda.nsight_systems
          pkgs.ffmpeg
          pkgs.curl
          python
        ];
        shellHook = ''
          # NixOS exposes the proprietary NVIDIA driver libraries, including libcuda.so.1,
          # under /run/opengl-driver/lib. Without that directory on LD_LIBRARY_PATH the
          # CUDA runtime cannot find the driver and reports the misleading
          # "CUDA driver version is insufficient for CUDA runtime version". Prepend it
          # once, leaving any existing LD_LIBRARY_PATH entries in place.
          case ":''${LD_LIBRARY_PATH:-}:" in
            *:/run/opengl-driver/lib:*) ;;
            *) export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
          esac
        '';
      };
    };
}
