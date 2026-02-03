import zipfile
import os
import shutil


class Zippify:
    # ----------------- EXISTENTES -----------------
    def comprimir(self, zip_path="", archivos=None, modo="w"):
        archivos = archivos or []
        with zipfile.ZipFile(zip_path, mode=modo, compression=zipfile.ZIP_DEFLATED) as zf:
            for archivo in archivos:
                if os.path.isfile(archivo):
                    zf.write(archivo, os.path.basename(archivo))
                else:
                    print(f"El archivo {archivo} no existe.")
        return zip_path

    def comprimir_carpeta(self, zip_path="", carpeta="", modo="w"):
        with zipfile.ZipFile(zip_path, mode=modo, compression=zipfile.ZIP_DEFLATED) as zf:
            for raiz, _, archivos in os.walk(carpeta):
                for archivo in archivos:
                    ruta_completa = os.path.join(raiz, archivo)
                    ruta_relativa = os.path.relpath(ruta_completa, start=carpeta)
                    zf.write(ruta_completa, ruta_relativa)
        return zip_path

    def descomprimir(self, zip_path="", destino="."):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(destino)

    def descomprimir_carpeta(self, zip_path="", destino=".", nombre_carpeta=None):
        if not zip_path:
            raise ValueError("Debe especificar la ruta del archivo ZIP.")

        destino_final = os.path.join(
            destino,
            nombre_carpeta or os.path.splitext(os.path.basename(zip_path))[0]
        )
        os.makedirs(destino_final, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(destino_final)
            carpetas = [info.filename.split('/')[0] for info in zf.infolist() if '/' in info.filename]
            carpetas_unicas = list(set(carpetas))

        if len(carpetas_unicas) == 1:
            carpeta_interna = os.path.join(destino_final, carpetas_unicas[0])
            return os.path.abspath(carpeta_interna)
        else:
            return os.path.abspath(destino_final)

    def listar_contenido(self, zip_path=""):
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.namelist()

    # ----------------- NUEVOS -----------------
    def comprimir_modelo_minimo(
        self,
        result_model_dir: str,
        zip_path: str | None = None,
        prefer: str = "keras",            # "keras" | "tflite"
        incluir_labels: bool = True
    ) -> str:
        """
        Empaqueta SOLO lo necesario para servir inferencia:
          - Keras:  model.keras | best.keras | last.keras
          - TFLite: model.tflite
          + labels.json (si incluir_labels=True)
        Los archivos quedan en la RAÍZ del ZIP.
        """
        result_model_dir = os.path.abspath(result_model_dir)
        if not os.path.isdir(result_model_dir):
            raise FileNotFoundError(f"No existe la carpeta: {result_model_dir}")

        # orden de preferencia
        keras_candidates = ["model.keras", "best.keras", "last.keras", "best_ft.keras", "best_head.keras"]
        tflite_candidates = ["model.tflite"]

        ordered = (keras_candidates + tflite_candidates) if prefer == "keras" else (tflite_candidates + keras_candidates)

        selected_model = None
        for name in ordered:
            p = os.path.join(result_model_dir, name)
            if os.path.isfile(p):
                selected_model = p
                break

        if not selected_model:
            raise FileNotFoundError(f"No se encontró modelo (.keras/.tflite) en {result_model_dir}")

        files_to_zip = [selected_model]

        if incluir_labels:
            labels_path = os.path.join(result_model_dir, "labels.json")
            if not os.path.isfile(labels_path):
                raise FileNotFoundError(f"labels.json no encontrado en {result_model_dir}")
            files_to_zip.append(labels_path)

        # destino
        if zip_path is None:
            zip_path = os.path.join(os.path.dirname(result_model_dir), "deploy_bundle.zip")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                zf.write(f, arcname=os.path.basename(f))

        return zip_path

    def comprimir_deploy(self, run_dir: str, zip_path: str | None = None) -> str:
        """
        Zipea la carpeta run_dir/result_model/deploy (si ya generas deploy/).
        """
        deploy_dir = os.path.join(run_dir, "result_model", "deploy")
        if not os.path.isdir(deploy_dir):
            raise FileNotFoundError(f"No existe la carpeta deploy: {deploy_dir}")

        if zip_path is None:
            zip_path = os.path.join(run_dir, "deploy_bundle.zip")

        return self.comprimir_carpeta(zip_path=zip_path, carpeta=deploy_dir)

    # (Opcional) borrar lo no necesario antes de subir
    def eliminar_no_necesarios(self, result_model_dir: str, keep: list[str]) -> None:
        """
        Elimina TODO lo que no esté en 'keep' dentro de result_model_dir.
        'keep' debe ser nombres de archivo relativos, p.ej. ["model.keras", "labels.json"].
        """
        result_model_dir = os.path.abspath(result_model_dir)
        for nombre in os.listdir(result_model_dir):
            ruta = os.path.join(result_model_dir, nombre)
            if os.path.isdir(ruta):
                if nombre not in keep:
                    shutil.rmtree(ruta, ignore_errors=True)
            else:
                if nombre not in keep:
                    try:
                        os.remove(ruta)
                    except OSError:
                        pass
