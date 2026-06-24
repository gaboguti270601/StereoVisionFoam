%% CAPTURA ESTÉREO THORLABS ZELUX EN MATLAB
% Usa el SDK .NET oficial de Thorlabs, no Image Acquisition Toolbox.
%
% Funciones:
% 1) abre dos cámaras por número de serie;
% 2) configura exposición;
% 3) captura pares de frames;
% 4) guarda un MP4 por cámara;
% 5) usa NOMBRE_EXPERIENCIA en todos los archivos;
% 6) guarda CSV con timestamp e intervalo real entre pares;
% 7) guarda metadata de la experiencia;
% 8) termina con Q, Escape o cerrando la ventana.
%
% Requisitos:
% - MATLAB de 64 bits en Windows
% - ThorCam / Scientific Camera Support instalado
% - Thorlabs.TSI.TLCamera.dll y dependencias disponibles
%
% La API usada corresponde al SDK .NET de Thorlabs:
%   TLCameraSDK.OpenTLCameraSDK
%   DiscoverAvailableCameras
%   OpenCamera
%   ExposureTime_us
%   Arm
%   IssueSoftwareTrigger
%   GetPendingFrameOrNull

clear;
clc;
close all;

%% ============================================================
% CONFIGURACIÓN DEL USUARIO
% =============================================================

NOMBRE_EXPERIENCIA = "II38P";

CARPETA_BASE = "D:\MDT\Stereovision";

%CAMARA_L_SERIAL = "36933";
CAMARA_L_SERIAL = "28327";
CAMARA_R_SERIAL = "36930";

EXPOSICION_US = 40;

FPS_OBJETIVO = 30.0;
FRAME_TIME = 1.0 / FPS_OBJETIVO;

VIDEO_CODEC_DECLARADO = "mp4v";
PERFIL_VIDEO_MATLAB = "MPEG-4";
CALIDAD_VIDEO = 90;

% 0 = grabar hasta presionar Q/Escape o cerrar la ventana.
DURACION_MAXIMA_S = 0;

MOSTRAR_VISTA_PREVIA = true;
ESCALA_PREVIA = 0.5;

% Ruta de la DLL principal.
% Ajusta esta ruta según tu instalación.
RUTA_DLL_TLCAMERA = fullfile( ...
    "D:\MDT\Scientific Camera Interfaces", ...
    "SDK", ...
    "DotNet Toolkit", ...
    "dlls", ...
    "Managed_64_lib", ...
    "Thorlabs.TSI.TLCamera.dll" ...
);

% Tiempo máximo de espera por frame.
TIMEOUT_FRAME_S = 2.0;

%% ============================================================
% RUTAS DE SALIDA
% =============================================================

CARPETA_SALIDA = fullfile(CARPETA_BASE, NOMBRE_EXPERIENCIA);

if ~exist(CARPETA_SALIDA, 'dir')
    mkdir(CARPETA_SALIDA);
end

MARCA_INICIO = string(datetime('now', ...
    'Format', 'yyyy-MM-dd_HH-mm-ss'));

RUTA_VIDEO_L = fullfile( ...
    CARPETA_SALIDA, ...
    NOMBRE_EXPERIENCIA + "_cam1_" + ...
    CAMARA_L_SERIAL + "_" + MARCA_INICIO + ".mp4" ...
);

RUTA_VIDEO_R = fullfile( ...
    CARPETA_SALIDA, ...
    NOMBRE_EXPERIENCIA + "_cam2_" + ...
    CAMARA_R_SERIAL + "_" + MARCA_INICIO + ".mp4" ...
);

RUTA_REGISTRO = fullfile( ...
    CARPETA_SALIDA, ...
    NOMBRE_EXPERIENCIA + "_captura_" + ...
    MARCA_INICIO + ".csv" ...
);

RUTA_METADATA = fullfile( ...
    CARPETA_SALIDA, ...
    NOMBRE_EXPERIENCIA + "_metadata_" + ...
    MARCA_INICIO + ".csv" ...
);

%% ============================================================
% COMPROBAR DLL
% =============================================================

if ~isfile(RUTA_DLL_TLCAMERA)
    mensajeError = sprintf([ ...
        'No se encontró la DLL principal de Thorlabs en:\n%s\n\n' ...
        'Busca el archivo Thorlabs.TSI.TLCamera.dll y reemplaza ' ...
        'RUTA_DLL_TLCAMERA en la parte superior del código.' ...
    ], char(RUTA_DLL_TLCAMERA));

    error('%s', mensajeError);
end

% Agregar la carpeta de las DLL al PATH del proceso actual.
CARPETA_DLL = fileparts(RUTA_DLL_TLCAMERA);
setenv('PATH', CARPETA_DLL + ";" + string(getenv('PATH')));

%% ============================================================
% VARIABLES PARA CIERRE SEGURO
% =============================================================

tlCameraSDK = [];
cameraL = [];
cameraR = [];
serialNumbers = [];
writerL = [];
writerR = [];
archivoRegistro = -1;
figura = [];

try
    %% ========================================================
    % CARGAR SDK .NET
    % =========================================================

    NET.addAssembly(char(RUTA_DLL_TLCAMERA));

    fprintf('\nDLL cargada:\n%s\n', RUTA_DLL_TLCAMERA);

    tlCameraSDK = ...
        Thorlabs.TSI.TLCamera.TLCameraSDK.OpenTLCameraSDK;

    serialNumbers = tlCameraSDK.DiscoverAvailableCameras;

    fprintf('\n========================================\n');
    fprintf('CÁMARAS DETECTADAS\n');
    fprintf('========================================\n');
    fprintf('Cantidad: %d\n', serialNumbers.Count);

    serialesDetectados = strings(serialNumbers.Count, 1);

    for k = 0:(serialNumbers.Count - 1)
        serialActual = string(serialNumbers.Item(k));
        serialesDetectados(k + 1) = serialActual;
        fprintf('%d: %s\n', k + 1, serialActual);
    end

    if ~any(serialesDetectados == CAMARA_L_SERIAL)
        error('No se encontró la cámara izquierda %s.', ...
            CAMARA_L_SERIAL);
    end

    if ~any(serialesDetectados == CAMARA_R_SERIAL)
        error('No se encontró la cámara derecha %s.', ...
            CAMARA_R_SERIAL);
    end

    %% ========================================================
    % ABRIR CÁMARAS
    % =========================================================

    cameraL = tlCameraSDK.OpenCamera( ...
        char(CAMARA_L_SERIAL), ...
        false ...
    );

    cameraR = tlCameraSDK.OpenCamera( ...
        char(CAMARA_R_SERIAL), ...
        false ...
    );

    fprintf('\nCámara izquierda abierta: %s\n', CAMARA_L_SERIAL);
    fprintf('Cámara derecha abierta: %s\n', CAMARA_R_SERIAL);

    %% ========================================================
    % CONFIGURAR CÁMARAS
    % =========================================================

    configurarCamara(cameraL, EXPOSICION_US);
    configurarCamara(cameraR, EXPOSICION_US);

    bitDepthL = double(cameraL.BitDepth);
    bitDepthR = double(cameraR.BitDepth);

    fprintf('\n========================================\n');
    fprintf('CONFIGURACIÓN\n');
    fprintf('========================================\n');
    fprintf('Experiencia: %s\n', NOMBRE_EXPERIENCIA);
    fprintf('Exposición solicitada: %.3f us\n', EXPOSICION_US);
    fprintf('FPS objetivo: %.3f fps\n', FPS_OBJETIVO);
    fprintf('Frame time objetivo: %.9f s\n', FRAME_TIME);
    fprintf('Bit depth L: %.0f bits\n', bitDepthL);
    fprintf('Bit depth R: %.0f bits\n', bitDepthR);

    %% ========================================================
    % PREPARAR ADQUISICIÓN
    % =========================================================

    cameraL.OperationMode = ...
        Thorlabs.TSI.TLCameraInterfaces.OperationMode.SoftwareTriggered;

    cameraR.OperationMode = ...
        Thorlabs.TSI.TLCameraInterfaces.OperationMode.SoftwareTriggered;

    % Un frame por trigger para controlar los pares.
    cameraL.FramesPerTrigger_zeroForUnlimited = 1;
    cameraR.FramesPerTrigger_zeroForUnlimited = 1;

    cameraL.MaximumNumberOfFramesToQueue = 10;
    cameraR.MaximumNumberOfFramesToQueue = 10;

    cameraL.Arm;
    cameraR.Arm;

    % Capturar primer par para conocer resolución.
    [frameL16, infoFrameL] = capturarFrame( ...
        cameraL, ...
        TIMEOUT_FRAME_S ...
    );

    [frameR16, infoFrameR] = capturarFrame( ...
        cameraR, ...
        TIMEOUT_FRAME_S ...
    );

    if ~isequal(size(frameL16), size(frameR16))
        error([ ...
            'Las cámaras entregan resoluciones diferentes. ' ...
            'L=%dx%d, R=%dx%d.' ...
        ], ...
            size(frameL16, 2), size(frameL16, 1), ...
            size(frameR16, 2), size(frameR16, 1));
    end

    alto = size(frameL16, 1);
    ancho = size(frameL16, 2);

    frameL8 = convertirAUint8(frameL16, bitDepthL);
    frameR8 = convertirAUint8(frameR16, bitDepthR);

    %% ========================================================
    % CREAR VIDEOS
    % =========================================================

    writerL = VideoWriter( ...
        RUTA_VIDEO_L, ...
        PERFIL_VIDEO_MATLAB ...
    );

    writerR = VideoWriter( ...
        RUTA_VIDEO_R, ...
        PERFIL_VIDEO_MATLAB ...
    );

    writerL.FrameRate = FPS_OBJETIVO;
    writerR.FrameRate = FPS_OBJETIVO;

    if isprop(writerL, 'Quality')
        writerL.Quality = CALIDAD_VIDEO;
        writerR.Quality = CALIDAD_VIDEO;
    end

    open(writerL);
    open(writerR);

    %% ========================================================
    % CREAR CSV
    % =========================================================

    archivoRegistro = fopen( ...
        RUTA_REGISTRO, ...
        'w', ...
        'n', ...
        'UTF-8' ...
    );

    if archivoRegistro == -1
        error('No se pudo crear el CSV de registro.');
    end

    fprintf(archivoRegistro, [ ...
        'nombre_experiencia,' ...
        'frame_grabado,' ...
        'timestamp,' ...
        'timestamp_posix_s,' ...
        'tiempo_desde_inicio_s,' ...
        'intervalo_desde_frame_anterior_s,' ...
        'fps_instantaneo,' ...
        'fps_objetivo,' ...
        'frame_time_objetivo_s,' ...
        'exposicion_us,' ...
        'camara_l_serial,' ...
        'camara_r_serial,' ...
        'frame_sdk_l,' ...
        'frame_sdk_r,' ...
        'video_codec\n' ...
    ]);

    %% ========================================================
    % GUARDAR METADATA
    % =========================================================

    metadata = table( ...
        NOMBRE_EXPERIENCIA, ...
        MARCA_INICIO, ...
        CAMARA_L_SERIAL, ...
        CAMARA_R_SERIAL, ...
        EXPOSICION_US, ...
        FPS_OBJETIVO, ...
        FRAME_TIME, ...
        VIDEO_CODEC_DECLARADO, ...
        PERFIL_VIDEO_MATLAB, ...
        ancho, ...
        alto, ...
        bitDepthL, ...
        bitDepthR, ...
        string(RUTA_VIDEO_L), ...
        string(RUTA_VIDEO_R), ...
        string(RUTA_REGISTRO), ...
        string(RUTA_DLL_TLCAMERA), ...
        'VariableNames', { ...
            'nombre_experiencia', ...
            'marca_inicio', ...
            'camara_l_serial', ...
            'camara_r_serial', ...
            'exposicion_us', ...
            'fps_objetivo', ...
            'frame_time_objetivo_s', ...
            'video_codec_declarado', ...
            'perfil_video_matlab', ...
            'ancho_px', ...
            'alto_px', ...
            'bit_depth_l', ...
            'bit_depth_r', ...
            'ruta_video_l', ...
            'ruta_video_r', ...
            'ruta_registro', ...
            'ruta_dll_tlcamera' ...
        } ...
    );

    writetable(metadata, RUTA_METADATA);

    %% ========================================================
    % VISTA PREVIA
    % =========================================================

    if MOSTRAR_VISTA_PREVIA
        figura = figure( ...
            'Name', ...
            'Captura estéreo Thorlabs - Q/Escape para detener', ...
            'NumberTitle', ...
            'off', ...
            'Color', ...
            'black', ...
            'KeyPressFcn', ...
            @marcarDetencion, ...
            'CloseRequestFcn', ...
            @cerrarSolicitandoDetencion ...
        );

        setappdata(figura, 'detener', false);

        vista = prepararVistaPrevia( ...
            frameL8, ...
            frameR8, ...
            ESCALA_PREVIA ...
        );

        manejadorImagen = imshow(vista);

        titulo = title( ...
            NOMBRE_EXPERIENCIA + " | iniciando...", ...
            'Color', ...
            'white' ...
        );
    end

    %% ========================================================
    % CAPTURA CONTINUA
    % =========================================================

    fprintf('\n========================================\n');
    fprintf('CAPTURA INICIADA\n');
    fprintf('========================================\n');
    fprintf('Resolución: %d x %d\n', ancho, alto);
    fprintf('Video L:\n%s\n', RUTA_VIDEO_L);
    fprintf('Video R:\n%s\n', RUTA_VIDEO_R);
    fprintf('Registro:\n%s\n', RUTA_REGISTRO);
    fprintf('\nPresiona Q o Escape para detener.\n');

    inicio = tic;
    tiempoAnterior = NaN;
    numeroFrame = 0;

    % Guardar primer par ya capturado.
    continuar = true;
    usarPrimerPar = true;

    while continuar
        inicioIteracion = tic;

        if MOSTRAR_VISTA_PREVIA
            if ~isgraphics(figura)
                break;
            end

            if getappdata(figura, 'detener')
                break;
            end
        end

        if usarPrimerPar
            usarPrimerPar = false;
        else
            % Disparar ambas cámaras tan cerca como permite MATLAB.
            cameraL.IssueSoftwareTrigger;
            cameraR.IssueSoftwareTrigger;

            [frameL16, infoFrameL] = obtenerFramePendiente( ...
                cameraL, ...
                TIMEOUT_FRAME_S ...
            );

            [frameR16, infoFrameR] = obtenerFramePendiente( ...
                cameraR, ...
                TIMEOUT_FRAME_S ...
            );

            frameL8 = convertirAUint8(frameL16, bitDepthL);
            frameR8 = convertirAUint8(frameR16, bitDepthR);
        end

        writeVideo(writerL, frameL8);
        writeVideo(writerR, frameR8);

        tiempoActual = toc(inicio);

        fechaActual = datetime( ...
            'now', ...
            'Format', ...
            'yyyy-MM-dd HH:mm:ss.SSS' ...
        );

        if isnan(tiempoAnterior)
            intervaloReal = NaN;
            fpsInstantaneo = NaN;
        else
            intervaloReal = tiempoActual - tiempoAnterior;

            if intervaloReal > 0
                fpsInstantaneo = 1 / intervaloReal;
            else
                fpsInstantaneo = NaN;
            end
        end

        fprintf(archivoRegistro, ...
            '%s,%d,%s,%.6f,%.6f,', ...
            NOMBRE_EXPERIENCIA, ...
            numeroFrame, ...
            string(fechaActual), ...
            posixtime(fechaActual), ...
            tiempoActual ...
        );

        if isnan(intervaloReal)
            fprintf(archivoRegistro, ',,');
        else
            fprintf(archivoRegistro, ...
                '%.9f,%.3f,', ...
                intervaloReal, ...
                fpsInstantaneo ...
            );
        end

        fprintf(archivoRegistro, ...
            '%.3f,%.9f,%.3f,%s,%s,%d,%d,%s\n', ...
            FPS_OBJETIVO, ...
            FRAME_TIME, ...
            EXPOSICION_US, ...
            CAMARA_L_SERIAL, ...
            CAMARA_R_SERIAL, ...
            infoFrameL.frameNumber, ...
            infoFrameR.frameNumber, ...
            VIDEO_CODEC_DECLARADO ...
        );

        if mod(numeroFrame, 30) == 0
            fflush(archivoRegistro);
        end

        if MOSTRAR_VISTA_PREVIA && isgraphics(figura)
            vista = prepararVistaPrevia( ...
                frameL8, ...
                frameR8, ...
                ESCALA_PREVIA ...
            );

            manejadorImagen.CData = vista;

            titulo.String = sprintf( ...
                '%s | frame %d | %.1f s', ...
                NOMBRE_EXPERIENCIA, ...
                numeroFrame, ...
                tiempoActual ...
            );

            drawnow limitrate;
        end

        numeroFrame = numeroFrame + 1;
        tiempoAnterior = tiempoActual;

        if DURACION_MAXIMA_S > 0 && ...
                tiempoActual >= DURACION_MAXIMA_S
            fprintf('Se alcanzó la duración máxima.\n');
            continuar = false;
        end

        duracionIteracion = toc(inicioIteracion);
        espera = FRAME_TIME - duracionIteracion;

        if espera > 0
            pause(espera);
        end
    end

    %% ========================================================
    % CIERRE NORMAL
    % =========================================================

    fclose(archivoRegistro);
    archivoRegistro = -1;

    close(writerL);
    close(writerR);

    cameraL.Disarm;
    cameraR.Disarm;

    cameraL.Dispose;
    cameraR.Dispose;

    delete(cameraL);
    delete(cameraR);

    delete(serialNumbers);

    tlCameraSDK.Dispose;
    delete(tlCameraSDK);

    if isgraphics(figura)
        delete(figura);
    end

    fprintf('\n========================================\n');
    fprintf('CAPTURA FINALIZADA\n');
    fprintf('========================================\n');
    fprintf('Frames guardados por cámara: %d\n', numeroFrame);
    fprintf('Video izquierdo:\n%s\n', RUTA_VIDEO_L);
    fprintf('Video derecho:\n%s\n', RUTA_VIDEO_R);
    fprintf('Registro:\n%s\n', RUTA_REGISTRO);
    fprintf('Metadata:\n%s\n', RUTA_METADATA);

catch ME
    %% ========================================================
    % CIERRE SEGURO SI OCURRE UN ERROR
    % =========================================================

    if archivoRegistro ~= -1
        try
            fclose(archivoRegistro);
        catch
        end
    end

    if ~isempty(writerL)
        try
            close(writerL);
        catch
        end
    end

    if ~isempty(writerR)
        try
            close(writerR);
        catch
        end
    end

    if ~isempty(cameraL)
        try
            cameraL.Disarm;
        catch
        end

        try
            cameraL.Dispose;
        catch
        end

        try
            delete(cameraL);
        catch
        end
    end

    if ~isempty(cameraR)
        try
            cameraR.Disarm;
        catch
        end

        try
            cameraR.Dispose;
        catch
        end

        try
            delete(cameraR);
        catch
        end
    end

    if ~isempty(serialNumbers)
        try
            delete(serialNumbers);
        catch
        end
    end

    if ~isempty(tlCameraSDK)
        try
            tlCameraSDK.Dispose;
        catch
        end

        try
            delete(tlCameraSDK);
        catch
        end
    end

    if ~isempty(figura) && isgraphics(figura)
        delete(figura);
    end

    rethrow(ME);
end

%% ============================================================
% FUNCIONES LOCALES
% =============================================================

function configurarCamara(camera, exposicionUs)
    camera.ExposureTime_us = int64(round(exposicionUs));

    gainRange = camera.GainRange;

    if gainRange.Maximum > 0
        camera.Gain = 0;
    end

    camera.MaximumNumberOfFramesToQueue = 10;
end


function [imagen, info] = capturarFrame(camera, timeoutS)
    camera.IssueSoftwareTrigger;

    [imagen, info] = obtenerFramePendiente( ...
        camera, ...
        timeoutS ...
    );
end


function [imagen, info] = obtenerFramePendiente(camera, timeoutS)
    inicioEspera = tic;
    imageFrame = [];

    while isempty(imageFrame)
        imageFrame = camera.GetPendingFrameOrNull;

        if ~isempty(imageFrame)
            break;
        end

        if toc(inicioEspera) > timeoutS
            error('Tiempo de espera agotado al recibir un frame.');
        end

        pause(0.001);
    end

    try
        ancho = double(imageFrame.ImageData.Width_pixels);
        alto = double(imageFrame.ImageData.Height_pixels);
        frameNumber = double(imageFrame.FrameNumber);

        datosNet = ...
            imageFrame.ImageData.ImageData_monoOrBGR;

        datos = uint16(datosNet);

        imagen = reshape(datos, [ancho, alto])';

        info.width = ancho;
        info.height = alto;
        info.frameNumber = frameNumber;

    catch ME
        delete(imageFrame);
        rethrow(ME);
    end

    delete(imageFrame);
end


function imagen8 = convertirAUint8(imagen16, bitDepth)
    if isa(imagen16, 'uint8')
        imagen8 = imagen16;
        return;
    end

    desplazamiento = max(0, round(bitDepth) - 8);

    if desplazamiento > 0
        imagen8 = uint8(bitshift(imagen16, -desplazamiento));
    else
        imagen8 = uint8(imagen16);
    end
end


function vista = prepararVistaPrevia(frameL, frameR, escala)
    if escala ~= 1
        frameL = imresize(frameL, escala);
        frameR = imresize(frameR, escala);
    end

    separador = zeros(size(frameL, 1), 8, 'uint8');
    vista = [frameL, separador, frameR];
end


function marcarDetencion(figura, evento)
    if strcmpi(evento.Key, 'q') || ...
            strcmpi(evento.Key, 'escape')
        setappdata(figura, 'detener', true);
    end
end


function cerrarSolicitandoDetencion(figura, ~)
    setappdata(figura, 'detener', true);
end
