import json
import h5py

MODEL_PATH = "model/penguin_classifier.h5"

def patch_batch_shape_to_batch_input_shape(model_path: str):
    with h5py.File(model_path, "r+") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            raise RuntimeError("No se encontró 'model_config' en el .h5")

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        cfg = json.loads(model_config)

        # Navega: cfg["config"]["layers"][i]["config"]
        layers = cfg.get("config", {}).get("layers", [])
        changed = 0

        for layer in layers:
            layer_cfg = layer.get("config", {})
            if "batch_shape" in layer_cfg and "batch_input_shape" not in layer_cfg:
                layer_cfg["batch_input_shape"] = layer_cfg["batch_shape"]
                del layer_cfg["batch_shape"]
                changed += 1

        if changed == 0:
            print("No había 'batch_shape' que parchear. (Quizá tu modelo ya está bien)")
            return

        new_model_config = json.dumps(cfg).encode("utf-8")
        f.attrs.modify("model_config", new_model_config)

        print(f"✅ Parche aplicado. Capas modificadas: {changed}")

if __name__ == "__main__":
    patch_batch_shape_to_batch_input_shape(MODEL_PATH)
