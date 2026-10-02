"""Download and extract TTS candidate models from the sherpa-onnx release page."""
import argparse

from candidates import CANDIDATES, MODELS_ROOT, ensure_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", default=list(CANDIDATES), choices=list(CANDIDATES))
    args = parser.parse_args()
    for name in args.candidates:
        cand = CANDIDATES[name]
        print(f"{name}: languages={','.join(cand.languages)} licence={cand.licence}")
        print(f"  -> {ensure_model(cand, MODELS_ROOT)}")


if __name__ == "__main__":
    main()
