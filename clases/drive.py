import base64
import json
import os

import aiohttp


class Drive:
    def __init__(self, path_ws_rpa, api_key, modulo_id, id_folder_origen):
        self.urlconf = {
            "pathWsRPA": path_ws_rpa,
            "ak": api_key,
            "moduloId": modulo_id,
            "id_folder_origen": id_folder_origen
        }

    async def convert_base64(self, file_path):
        try:
            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return encoded
        except Exception as e:
            print(f"Error al convertir archivo a base64: {e}")
            raise

    async def get_module_drive(self):
        url = f"{self.urlconf['pathWsRPA']}/api/RPA/ModuloDriveCON"
        payload = {
            "ApiKey": self.urlconf["ak"],
            "Modulo_Id": self.urlconf["moduloId"]
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    print(data)
                    if "Root" in data and len(data["Root"]) > 0:
                        return data["Root"][0]["Id_ModuloDrive"]
                    return False
            except Exception as e:
                print(f"Error en get_module_drive: {e}")
                raise

    async def subir_drive(self, nombre_archivo, drive_destino, b64, mimetype):
        url = f"{self.urlconf['pathWsRPA']}/api/RPA/SubirArchivoV2"
        payload = {
            "ApiKey": self.urlconf["ak"],
            "ArchivoNombre": nombre_archivo,
            "IdFolderOrigen": drive_destino,
            "ArchivoBase64": b64,
            "MimeType": mimetype
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    raw = await resp.text()
                    data = json.loads(raw)
                    iddrive = data["Id"]
                    return iddrive
            except Exception as e:
                print(f"Error en subir_drive: {e}")
                return False

    async def descarga_drive(self, file_id):
        url = f"{self.urlconf['pathWsRPA']}/api/RPA/DescargarArchivo"
        payload = {
            "ApiKey": self.urlconf["ak"],
            "FileDriveId": file_id
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = json.loads(await resp.text())
                        return data.get("Archivo")
                    else:
                        print("Error al obtener el archivo")
                        raise Exception("Error al obtener el archivo")
            except Exception as e:
                print(f"Error en descarga_drive: {e}")
                raise

    async def descargar_a(self, adjunto):
        try:
            archivo_base64 = await self.descarga_drive(adjunto["Id_Drive"])
            return archivo_base64
        except Exception as e:
            print("Error al obtener el archivo", e)

    async def subir_a(self, file_path):
        try:
            nombre_archivo = file_path.split("/")[-1]
            mime_type = "application/zip"  # Ajusta si es necesario
            b64_data = await self.convert_base64(file_path)
            modulo_drive_id = await self.get_module_drive()

            if modulo_drive_id and modulo_drive_id != "-2":
                id_drive = await self.subir_drive(nombre_archivo, modulo_drive_id, b64_data, mime_type)
                return id_drive
            else:
                print("No se pudo subir el archivo")
                return ""
        except Exception as e:
            print(f"Error en subir_a: {e}")
            raise

    async def subir_a_v2(self, file_path, mime_type=""):
        try:
            nombre_archivo = os.path.basename(file_path)
            b64_data = await self.convert_base64(file_path)

            if self.urlconf["id_folder_origen"]:
                id_drive = await self.subir_drive(nombre_archivo, self.urlconf["id_folder_origen"], b64_data, mime_type)
                return id_drive
            else:
                print("No se pudo subir el archivo")
                return ""
        except Exception as e:
            print(f"Error en subir_a: {e}")
            raise
