import nbformat as nbf
from pathlib import Path
from textwrap import dedent


def make_images_to_half_life(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Images → Foam half-life (baseline)
                Quick baseline that pairs the converted foam stacks with tabular features to predict the foam half-life (as defined in the converter: first time the height or BC hits half the max).
                - Loads NSID `.h5` files from `../data/foam`
                - Builds a compact feature table (bubble metrics + image summary stats)
                - Trains a scikit-learn random forest regressor
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path

                DATA_DIR = Path("../data/foam").resolve()
                h5_files = sorted(DATA_DIR.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError("No .h5 files in ../data/foam. Run scripts/convert_foam_to_h5.py first.")

                # Pick one dataset to start; switch index if desired
                H5_PATH = h5_files[0]
                print("Using", H5_PATH)
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                import numpy as np
                import pandas as pd
                from PIL import Image
                from SciFiReaders import NSIDReader
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_error

                # Load NSID content
                reader = NSIDReader(H5_PATH)
                datasets = reader.read()
                im_ds = datasets["Channel_000"]
                feat_ds = datasets["Channel_001"]

                feature_names = [
                    f.decode() if isinstance(f, (bytes, bytearray)) else f
                    for f in feat_ds.metadata.get("feature_names", [])
                ]
                X = pd.DataFrame(feat_ds.compute(), columns=feature_names)

                images = im_ds.compute().astype(np.float32) / 255.0
                # Lightweight image summary stats
                flat = images.reshape(images.shape[0], -1)
                X["intensity_mean"] = flat.mean(axis=1)
                X["intensity_std"] = flat.std(axis=1)

                label_val = float(im_ds.original_metadata.get("half_life_s", np.nan))
                y = np.full(len(X), label_val, dtype=np.float32)

                print(f"Frames: {len(X)}, Pixel size (mm): {im_ds.original_metadata.get('pixel_size_mm')}")
                print("Half-life target (every frame uses the dataset value):", label_val)
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Train/test split + model
                train_cols = [c for c in X.columns if c != "half_life_s"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X[train_cols], y, test_size=0.2, random_state=42
                )

                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("rf", RandomForestRegressor(n_estimators=150, random_state=42)),
                    ]
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                print(f"MAE (s): {mae:.3f}")
                print("Dataset-wide predicted half-life (mean over frames):", model.predict(X[train_cols]).mean())
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                import matplotlib.pyplot as plt

                rf = model.named_steps["rf"]
                importances = rf.feature_importances_
                order = np.argsort(importances)[::-1][:10]

                plt.barh(range(len(order)), importances[order][::-1])
                plt.yticks(range(len(order)), [train_cols[i] for i in order][::-1])
                plt.xlabel("Feature importance")
                plt.title("Top predictors of half-life")
                plt.show()
                """
            )
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def make_params_to_images(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Parameters → Image (baseline)
                Baseline conditional image predictor using a fast, non-deep-learning model:
                - Loads the foam NSID files
                - Picks a subset of tabular features (time, heights/volumes, lamella thickness) as inputs
                - Downsamples frames to 64×64
                - Uses a kNN regressor to predict images from parameters
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path

                DATA_DIR = Path("../data/foam").resolve()
                h5_files = sorted(DATA_DIR.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError("No .h5 files in ../data/foam. Run scripts/convert_foam_to_h5.py first.")

                H5_PATH = h5_files[0]
                print("Using", H5_PATH)
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                import numpy as np
                import pandas as pd
                from PIL import Image
                from SciFiReaders import NSIDReader
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.metrics import mean_squared_error

                reader = NSIDReader(H5_PATH)
                datasets = reader.read()
                im_ds = datasets["Channel_000"]
                feat_ds = datasets["Channel_001"]

                feature_names = [
                    f.decode() if isinstance(f, (bytes, bytearray)) else f
                    for f in feat_ds.metadata.get("feature_names", [])
                ]
                feat_df = pd.DataFrame(feat_ds.compute(), columns=feature_names)

                # Candidate input features (drop missing ones gracefully)
                preferred_cols = [
                    "t [s]",
                    "hfoam [mm]",
                    "hliquid [mm]",
                    "Vfoam [mL]",
                    "Vliquid [mL]",
                    "BC [mm⁻²]",
                    "lamella_thickness_mm",
                ]
                input_cols = [c for c in preferred_cols if c in feat_df.columns]
                X = feat_df[input_cols].copy()

                frames = im_ds.compute().astype(np.float32)
                target_size = 64
                def downsample_frame(frame: np.ndarray) -> np.ndarray:
                    img = Image.fromarray(frame.astype(np.uint8))
                    img = img.resize((target_size, target_size))
                    return np.array(img, dtype=np.float32) / 255.0

                frames_to_use = min(len(frames), 300)  # keep it light
                small_frames = np.stack([downsample_frame(frames[i]) for i in range(frames_to_use)])
                X = X.iloc[:frames_to_use]

                y = small_frames.reshape(frames_to_use, -1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
                    ]
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                print(f"MSE on held-out frames: {mse:.6f}")
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                import matplotlib.pyplot as plt

                # Visualize a few predictions
                n_show = 3
                for i in range(n_show):
                    gt = y_test[i].reshape(target_size, target_size)
                    pred = preds[i].reshape(target_size, target_size)
                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    axes[0].imshow(gt, cmap="gray")
                    axes[0].set_title("Ground truth")
                    axes[0].axis("off")
                    axes[1].imshow(pred, cmap="gray")
                    axes[1].set_title("Predicted")
                    axes[1].axis("off")
                    plt.show()
                """
            )
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main() -> None:
    base = Path(__file__).parent
    make_images_to_half_life(base / "foam_images_to_half_life.ipynb")
    make_params_to_images(base / "foam_parameters_to_image.ipynb")
    print("Notebooks written to", base)


if __name__ == "__main__":
    main()
