import argparse
import asyncio
import base64
import json
import logging
import os
import shutil
import time
import warnings
import aiohttp
import urllib3

from logging.handlers import RotatingFileHandler

from clases.drive import Drive
from clases.wsrpa import RPA
from clases.zippify import Zippify
from configuracion.config_bdd import get_credentials

from trainer.config import DataConfig, PreprocessConfig, CVConfig, TrainConfig, SystemConfig
from trainer.train import train_system

# ==========================
# CONFIGURACIÓN DE LOGGING
# ==========================
LOG_DIR = "logs"
LOG_FILE = "servicio_trainer.log"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, LOG_FILE)


handler = RotatingFileHandler(
    log_path,
    maxBytes=500 * 1024 * 1024,  # 500 MB
    backupCount=1,               # Mantener solo 1 backup
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        handler,
        logging.StreamHandler()
    ]
)

logging.info("Logger inicializado correctamente.")


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*tf.function retracing.*")

diccionario_config = get_credentials()
ws_rpa = diccionario_config.get("ws_rpa", "http://172.25.111.102:9061") #"http://localhost:55881"
ws_rpa_drive = diccionario_config.get("ws_rpa_drive", "http://172.25.111.102:9192")
apikey = diccionario_config.get("apikey", "")
idfolderdrive = diccionario_config.get("idfolderdrive", "")
contador_limite = diccionario_config.get("contador_limite", 100)

output_dir = "outputs/"
datasets_dir = "datasets"

zippify = Zippify()
uploader = Drive(
    path_ws_rpa=ws_rpa_drive,
    api_key=apikey,
    modulo_id="27",
    id_folder_origen=idfolderdrive
)

rpa = RPA(path_ws_rpa=ws_rpa)


def parse_args():
    ap = argparse.ArgumentParser(description="Generic ML Trainer (classification/regression) — v2")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["classification", "regression"], default=None)
    ap.add_argument("--mode", choices=["auto", "custom"], default="auto")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--search-iters", type=int, default=30)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--select-k", type=int, default=None)
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--ensemble-top-n", type=int, default=2)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    return ap.parse_args()


async def upload_file_in_chunks(file_path, api_key, id_folder_origen, mime_type, url="", chunk_size=5*1024*1024):

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size // chunk_size) + (1 if file_size % chunk_size else 0)

    logging.info(f"Subiendo archivo: {file_name}")
    logging.info(f"Tamaño total: {file_size} bytes")
    logging.info(f"Total de chunks: {total_chunks}")

    with open(file_path, "rb") as f:
        for chunk_number in range(1, total_chunks + 1):

            chunk_data = f.read(chunk_size)

            files = {
                "Chunk": (file_name, chunk_data, mime_type)
            }

            data = {
                "ApiKey": api_key,
                "FileName": file_name,
                "ChunkNumber": chunk_number,
                "TotalChunks": total_chunks,
                "IdFolderOrigen": id_folder_origen,
                "MimeType": mime_type
            }

            logging.info(f"Enviando chunk {chunk_number}/{total_chunks}")

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(url, json=data) as resp:
                        raw = await resp.text()
                        data = json.loads(raw)
                        iddrive = data["Id"]
                        return iddrive
                except Exception as e:
                    print(f"Error en subir_drive: {e}")
                    return False

    print("\nArchivo enviado completamente.")


async def upload(filepath="", mime=""):
    try:
        drive_id = await uploader.subir_a_v2(filepath, mime_type=mime)
        logging.info(f"Archivo subido con ID: {drive_id}")
        return drive_id
    except Exception as e:
        logging.error(f"Error en upload: {e}")
        return False


async def download(iddrive, filename):
    try:
        if iddrive:
            file = {"Id_Drive": iddrive, "MimeType": "application/zip"}

            archivo_b64 = await uploader.descargar_a(file)

            carpeta_destino = datasets_dir
            os.makedirs(carpeta_destino, exist_ok=True)

            file_path = os.path.join(carpeta_destino, filename)

            with open(file_path, "wb") as f:
                f.write(base64.b64decode(archivo_b64))

            logging.info(f"Archivo guardado en: {file_path}")
            return True, file_path
        else:
            return False
    except Exception as e:
        logging.error(f"Error en download: {e}")
        return False


def trainer(dataset="", out_put="", id_version=0):
    try:
        params = {
            "csv_path": dataset,
            "target": "Survived",
            "task": None,
            "mode": "auto",
            "models": None,
            "features": None,
            "folds": 5,
            "search_iters": 30,
            "n_jobs": 1,
            "class_weight": "balanced",
            "no_calibrate": "",
            "select_k": None,
            "sample_rows": None,
            "output": out_put,
            "ensemble_top_n": 2,
            "no_gpu": ""
        }
        if id_version != 0:
            config_list = asyncio.run(rpa.configuraciones_con(version_id=id_version, activo=1))
            if config_list:
                mapping = {
                    "target": "target",
                    "task": "task",
                    "mode": "mode",
                    "models": "models",
                    "folds": "folds",
                    "class-weight": "class_weight",
                    "n_jobs": "n_jobs",
                }
                for item in config_list:
                    nombre = item["nombre"]
                    if nombre in mapping:
                        key = mapping[nombre]

                        # Convertir numéricos cuando corresponde
                        valor = item["valor"]
                        valor = valor.split(";")[1] if ";" in valor else valor
                        if valor.isdigit() or (valor.startswith('-') and valor[1:].isdigit()):
                            valor = int(valor)

                        params[key] = valor
        print(params)
        base_out_dir = "outputs"
        # args = parse_args()

        data_cfg = DataConfig(
            csv_path=params["csv_path"], target=params["target"], features=params["features"],
            sample_rows=params["sample_rows"]
        )
        pre_cfg = PreprocessConfig(select_k_best=params["select_k"])
        cv_cfg = CVConfig(folds=params["folds"], n_iter_search=params["search_iters"],
                          n_jobs=params["n_jobs"])
        train_cfg = TrainConfig(
            task=params["task"], mode=params["mode"], model_names=params["models"],
            class_weight=params["class_weight"], calibrate=not params["no_calibrate"],
            output_dir=params["output"], ensemble_top_n=params["ensemble_top_n"],
            use_gpu_if_available=not params["no_gpu"],

        )
        sys_cfg = SystemConfig(data=data_cfg, preprocess=pre_cfg, cv=cv_cfg, train=train_cfg)
        report = train_system(sys_cfg)
        result_out_dir = report["run_dir"]
        return 0, json.dumps(report, indent=2, ensure_ascii=False), result_out_dir
    except Exception as e:
        logging.error(f"Error en trainer: {e}")
        return 1, e, ""


def procesa_pendiente(registro):
    result = {
        "error": 0,
        "detalle": ""
    }
    try:
        id_drive = str(registro["driveCSV"])
        nombre = registro["nombre"]
        id_version = registro["version_Id"]
        report = ""
        logging.info(f"Procesando versión: {id_version}")

        file_name = f"dataset_{id_version}.csv"

        if id_drive:
            download_ok, dataset_path = asyncio.run(download(iddrive=id_drive, filename=file_name))
            if download_ok and dataset_path:
                asyncio.run(rpa.pendientes_act(id_pendiente=registro["id_Pendiente"], estatus_id=3, activo=1,
                                               detalle=f"Entrenando..."))
                ret, report, modelpath = trainer(dataset=dataset_path, out_put=output_dir + nombre, id_version=id_version)
                if ret == 0:
                    if os.path.exists(f"{modelpath}/confusion_matrix.png"):
                        matrizconfu = f"{modelpath}/confusion_matrix.png"
                    elif f"{modelpath}/corr_heatmap.png":
                        matrizconfu = f"{modelpath}/corr_heatmap.png"

                    modelpathzip = zippify.comprimir_carpeta(zip_path=f"{modelpath}.zip",
                                                             carpeta=modelpath)
                    if os.path.exists(modelpathzip):
                        iddrive_modelpathzip = asyncio.run(upload(modelpathzip, "application/zip"))
                        iddrive_matrizconfu = asyncio.run(upload(matrizconfu, "image/png"))
                        asyncio.run(rpa.versiones_act(id_version=id_version, id_driveresultado=iddrive_modelpathzip,
                                                      id_drivematriz=iddrive_matrizconfu, report=report))
                        asyncio.run(
                            rpa.pendientes_act(id_pendiente=registro["id_Pendiente"], estatus_id=4, activo=1,
                                               detalle="ok"))
                    else:
                        asyncio.run(
                            rpa.pendientes_act(id_pendiente=registro["id_Pendiente"], estatus_id=5, activo=1,
                                               detalle=f"No se ha generado el archivo zip del modelo generado"))
                else:
                    asyncio.run(rpa.pendientes_act(id_pendiente=registro["id_Pendiente"], estatus_id=5, activo=1,
                                                   detalle=f"El entrenamiento finalizo con errores, detalle: {report}"))
                    result["error"] = 1
                    result["detalle"] = report

            else:
                result["error"] = 1
                result["detalle"] = "Error al descargar el dataSet"
                asyncio.run(rpa.pendientes_act(id_pendiente=registro["id_Pendiente"], estatus_id=5, activo=1,
                                               detalle=result["detalle"]))

        return result

    except Exception as e:
        logging.error(f"Error al procesar pendiente: {e}")
        result["error"] = 1
        result["detalle"] = str(e)
        return result


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    logging.info("Servicio iniciado.")
    contador_ciclos = 0
    try:
        shutil.rmtree("datasets")
        shutil.rmtree("outputs")
        logging.info("Carpetas temporales eliminadas.")
    except Exception as e:
        logging.warning(f"No se pudieron limpiar carpetas: {e}")

    except FileNotFoundError:
        print(f"Error: La carpeta especificada no existe")
        logging.info(f"Error: La carpeta especificada no existe")
    except OSError as e:
        print(f"Error al eliminar las carpetas de recursos temporales: {e}")
        logging.info(f"Error al eliminar las carpetas de recursos temporales: {e}")
    while True:
        try:
            bandeja_pendientes = asyncio.run(rpa.pendientes_con(activo=1))
            pendientes_count = len(bandeja_pendientes)
            contador_ciclos += 1
            if pendientes_count > 0:
                logging.info(f"Lista pendientes: {bandeja_pendientes}")
                logging.info("Pendientes actualizados a estatus=2 en espera.")

                # Actualizar estatus a 2
                for registro in bandeja_pendientes:
                    asyncio.run(rpa.pendientes_act(
                        id_pendiente=registro["id_Pendiente"],
                        estatus_id=2,
                        activo=1,
                        detalle=f"En cola de espera para procesar"
                    ))

                # Procesar cada pendiente
                for registro in bandeja_pendientes:
                    logging.info("************************************")
                    logging.info("Registro en proceso")
                    logging.info(registro)
                    res = procesa_pendiente(registro)
                    logging.info(f"Resultado del procesamiento: {res}")
                    logging.info("************************************")

            elif contador_ciclos == contador_limite:
                print("Sin pendientes.")
                contador_ciclos = 0
            # Espera antes de la siguiente iteración

        except Exception as e:
            logging.exception("Ocurrió un error inesperado:")


if __name__ == "__main__":
    FILE_PATH = "C:/Repos/botPredictor/testchunks.rar"
    API_KEY = "R2oRvCD7ru8"
    FOLDER_ID = "1vxr_WfYIYOZ6FQPx-bWj7st7es03AP-8"
    MIME_TYPE = "application/zip"
    API_URL = "https://localhost:7166/api/RPA/SubirArchivoV3"

    main()
    """
    asyncio.run(upload_file_in_chunks(
        file_path=FILE_PATH,
        api_key=API_KEY,
        id_folder_origen=FOLDER_ID,
        mime_type=MIME_TYPE,
        url=API_URL))
    """
