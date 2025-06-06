# Configuración
$csvPath = "urls.csv"             # Ruta al archivo CSV
$columnaURL = "url"               # Nombre de la columna que contiene las URLs
$destino = "pdfs"                 # Carpeta donde guardar los archivos

# Crear carpeta si no existe
if (-not (Test-Path -Path $destino)) {
    New-Item -ItemType Directory -Path $destino | Out-Null
}

# Leer y procesar cada URL
$urls = Import-Csv -Path $csvPath

$i = 1
foreach ($fila in $urls) {
    $url = $fila.$columnaURL
    if (![string]::IsNullOrWhiteSpace($url)) {
        Write-Host "[$i] Descargando: $url"
        try {
            # Obtener el nombre del archivo desde la URL
            $nombreArchivo = [System.Web.HttpUtility]::UrlDecode((Split-Path -Path $url -Leaf))
            $rutaDestino = Join-Path $destino $nombreArchivo

            # Descargar el PDF
            Invoke-WebRequest -Uri $url -OutFile $rutaDestino -TimeoutSec 30

            Write-Host "    ✅ Guardado como: $rutaDestino"
        } catch {
            Write-Host "    ⚠️ Error al descargar $url':' $_"
        }
        $i++
    }
}
