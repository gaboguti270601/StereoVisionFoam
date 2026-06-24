%% ESTIMACION DE ALTURA DE ESPUMA CON ESTEREOVISION EN MATLAB
% Requisitos:
% - MATLAB R2025b
% - Computer Vision Toolbox
% - Image Processing Toolbox
%
% Archivos esperados:
%   stereoParams.mat
%   II38P_cam1.mp4
%   II38P_cam2.mp4
%   II38P_captura.csv
%
% El programa:
% 1) carga la calibración estéreo;
% 2) rectifica los frames directamente con rectifyStereoImages;
% 3) permite verificar la rectificación;
% 4) solicita un punto correspondiente para estimar el rango de disparidad;
% 5) permite seleccionar la ROI de la espuma;
% 6) procesa ventanas temporales alrededor de las alturas conocidas;
% 7) ajusta modelos disparidad-altura;
% 8) guarda tabla, modelo y gráfico.

clear;
clc;
close all;

%% ============================================================
% CONFIGURACIÓN
% =============================================================

CARPETA_BASE = 'D:\MDT\Stereovision';

ARCHIVO_CALIBRACION = fullfile(CARPETA_BASE, 'stereoParams.mat');
VIDEO_L = fullfile(CARPETA_BASE, 'II38P_cam1.mp4');
VIDEO_R = fullfile(CARPETA_BASE, 'II38P_cam2.mp4');
CSV_REGISTRO = fullfile(CARPETA_BASE, 'II38P_captura.csv');

CARPETA_SALIDA = fullfile( ...
    CARPETA_BASE, ...
    'calibracion_empirica_MATLAB_II38P' ...
);

if ~exist(CARPETA_SALIDA, 'dir')
    mkdir(CARPETA_SALIDA);
end

% Frame utilizado para verificar rectificación y seleccionar ROI.
FRAME_PRUEBA = 500;

% Las horas manuales solo tienen precisión de minuto.
VENTANA_TEMPORAL_S = 30;

% Procesar un frame por segundo aproximadamente.
% Cambiar a 3 entrega más muestras, pero será mucho más lento.
SALTO_FRAMES_VENTANA = 30;

% El programa compensa primero la gran separación horizontal entre las
% imágenes rectificadas y luego calcula una disparidad residual en [-64, 64].

% Preprocesamiento
GAMMA_CORRECCION = 1.5;
CLAHE_CLIP_LIMIT = 0.012;
CLAHE_NUM_TILES = [8 8];
BILATERAL_DEGREE = 0.05;
BILATERAL_SPATIAL_SIGMA = 3;

% Máscara de bordes
RADIO_DILATACION_BORDES = 3;

% Filtros de validez
INTENSIDAD_MIN_VALIDA = 8;
INTENSIDAD_MAX_VALIDA = 245;
MIN_PIXELES_VALIDOS_FRAME = 80;
IQR_FACTOR = 1.5;

% Mediana de disparidad dentro de la ROI
PERCENTIL_DISPARIDAD_ROI = 50;

% Si la ROI cambia durante el ensayo, dejar true para elegirla manualmente.
SELECCIONAR_ROI = true;

% Solo se usa cuando SELECCIONAR_ROI = false.
ROI_FIJA = [1 1 300 300];  % [x y ancho alto]

% Mediciones manuales
MEDICIONES = table( ...
    datetime([ ...
        "2026-06-17 19:40:00.000"
        "2026-06-17 19:43:00.000"
        "2026-06-17 19:52:00.000"
        "2026-06-17 20:10:00.000"
        "2026-06-17 20:20:00.000"
        "2026-06-17 20:34:00.000"
        "2026-06-17 20:50:00.000" ...
    ], 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS'), ...
    [14; 12; 29; 37; 38; 45; 50], ...
    ["calibracion"; "calibracion"; "calibracion"; ...
     "calibracion"; "calibracion"; "validacion"; "validacion"], ...
    'VariableNames', {'timestamp', 'altura_mm', 'uso'} ...
);

%% ============================================================
% COMPROBACIÓN DE ARCHIVOS
% =============================================================

archivos = {
    ARCHIVO_CALIBRACION
    VIDEO_L
    VIDEO_R
    CSV_REGISTRO
};

for k = 1:numel(archivos)
    if ~isfile(archivos{k})
        error('No se encontró el archivo: %s', archivos{k});
    end
end

%% ============================================================
% CARGAR CALIBRACIÓN, VIDEOS Y CSV
% =============================================================

datosCalibracion = load(ARCHIVO_CALIBRACION, 'stereoParams');

if ~isfield(datosCalibracion, 'stereoParams')
    error('El archivo no contiene la variable stereoParams.');
end

stereoParams = datosCalibracion.stereoParams;

videoL = VideoReader(VIDEO_L);
videoR = VideoReader(VIDEO_R);

fprintf('\n========================================\n');
fprintf('VIDEOS\n');
fprintf('========================================\n');
fprintf('Cámara 1: %d x %d, %.3f fps\n', ...
    videoL.Width, videoL.Height, videoL.FrameRate);
fprintf('Cámara 2: %d x %d, %.3f fps\n', ...
    videoR.Width, videoR.Height, videoR.FrameRate);

if videoL.Width ~= videoR.Width || videoL.Height ~= videoR.Height
    error('Los dos videos tienen resoluciones diferentes.');
end

tamCal = stereoParams.CameraParameters1.ImageSize; % [alto ancho]

fprintf('\n========================================\n');
fprintf('CALIBRACIÓN\n');
fprintf('========================================\n');
fprintf('Tamaño calibración: %d filas x %d columnas\n', tamCal(1), tamCal(2));
fprintf('Error medio de reproyección: %.4f px\n', ...
    stereoParams.MeanReprojectionError);

registro = readtable(CSV_REGISTRO, 'VariableNamingRule', 'preserve');

if ~ismember('frame_grabado', registro.Properties.VariableNames)
    error('El CSV no contiene la columna frame_grabado.');
end

if ~ismember('timestamp', registro.Properties.VariableNames)
    error('El CSV no contiene la columna timestamp.');
end

if ~isdatetime(registro.timestamp)
    try
        registro.timestamp = datetime( ...
            string(registro.timestamp), ...
            'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS' ...
        );
    catch
        registro.timestamp = datetime(string(registro.timestamp));
    end
end

registro = rmmissing(registro, 'DataVariables', 'timestamp');
registro = sortrows(registro, 'timestamp');

if isempty(registro)
    error('El CSV no contiene timestamps válidos.');
end

%% ============================================================
% FRAME DE PRUEBA Y RECTIFICACIÓN
% =============================================================

nFramesEstimadoL = floor(videoL.Duration * videoL.FrameRate);
nFramesEstimadoR = floor(videoR.Duration * videoR.FrameRate);
frameMaximo = min(nFramesEstimadoL, nFramesEstimadoR);

FRAME_PRUEBA = max(1, min(FRAME_PRUEBA, frameMaximo));

frameL = read(videoL, FRAME_PRUEBA);
frameR = read(videoR, FRAME_PRUEBA);

[frameLcal, frameRcal] = adaptarResolucionCalibracion( ...
    frameL, frameR, tamCal ...
);

[rectL, rectR] = rectifyStereoImages( ...
    frameLcal, ...
    frameRcal, ...
    stereoParams, ...
    'OutputView', 'full' ...
);

verificarRectificacion(rectL, rectR);

%% ============================================================
% ESTIMAR RANGO DE DISPARIDAD MEDIANTE UN PUNTO CORRESPONDIENTE
% =============================================================

[desplazamientoHorizontalR, disparidadPunto] = ...
    seleccionarAlineacionHorizontal(rectL, rectR);

% Desplazar la imagen derecha para llevar la disparidad grande cerca de cero.
rectR = desplazarHorizontal(rectR, desplazamientoHorizontalR);

% Después de la compensación se busca disparidad residual en ±64 px.
minDisparity = -64;
maxDisparity = 64;

fprintf('\n========================================\n');
fprintf('ALINEACIÓN HORIZONTAL Y DISPARIDAD\n');
fprintf('========================================\n');
fprintf('Disparidad original del punto: %.3f px\n', disparidadPunto);
fprintf('Desplazamiento aplicado a cámara 2: %.3f px\n', ...
    desplazamientoHorizontalR);
fprintf('Rango residual disparitySGM: [%d, %d] px\n', ...
    minDisparity, maxDisparity);

%% ============================================================
% SELECCIÓN DE ROI
% =============================================================

primeraHora = MEDICIONES.timestamp(1);
coincidenciaInicial = buscarFrameMasCercano(registro, primeraHora);

frameCentral = coincidenciaInicial.frame_video_matlab;

frameLroi = read(videoL, frameCentral);
frameRroi = read(videoR, frameCentral);

[frameLroi, frameRroi] = adaptarResolucionCalibracion( ...
    frameLroi, frameRroi, tamCal ...
);

[rectLroi, rectRroi] = rectifyStereoImages( ...
    frameLroi, ...
    frameRroi, ...
    stereoParams, ...
    'OutputView', 'full' ...
);

rectRroi = desplazarHorizontal( ...
    rectRroi, ...
    desplazamientoHorizontalR ...
);

if SELECCIONAR_ROI
    ROI = seleccionarROI(rectLroi);
else
    ROI = ROI_FIJA;
end

fprintf('\nROI seleccionada: [x=%d, y=%d, ancho=%d, alto=%d]\n', ...
    ROI(1), ROI(2), ROI(3), ROI(4));

%% ============================================================
% PRUEBA DE DISPARIDAD SOBRE EL FRAME CENTRAL
% =============================================================

[dispPrueba, edgeMaskPrueba, grayLPrueba] = calcularDisparidadFrame( ...
    rectLroi, ...
    rectRroi, ...
    minDisparity, ...
    maxDisparity, ...
    GAMMA_CORRECCION, ...
    CLAHE_CLIP_LIMIT, ...
    CLAHE_NUM_TILES, ...
    BILATERAL_DEGREE, ...
    BILATERAL_SPATIAL_SIGMA, ...
    RADIO_DILATACION_BORDES, ...
    INTENSIDAD_MIN_VALIDA, ...
    INTENSIDAD_MAX_VALIDA ...
);

mostrarDiagnosticoDisparidad( ...
    rectLroi, ...
    dispPrueba, ...
    edgeMaskPrueba, ...
    grayLPrueba, ...
    ROI, ...
    minDisparity, ...
    maxDisparity ...
);

respuesta = questdlg( ...
    '¿La ROI y el mapa de disparidad se ven razonables?', ...
    'Confirmar procesamiento', ...
    'Sí, continuar', ...
    'No, detener', ...
    'Sí, continuar' ...
);

if ~strcmp(respuesta, 'Sí, continuar')
    error('Procesamiento cancelado por el usuario.');
end

%% ============================================================
% PROCESAR TODAS LAS MEDICIONES
% =============================================================

nMediciones = height(MEDICIONES);

timestamp_solicitado = NaT(nMediciones, 1);
timestamp_encontrado = NaT(nMediciones, 1);
diferencia_temporal_s = nan(nMediciones, 1);
frame_central = nan(nMediciones, 1);
altura_mm = MEDICIONES.altura_mm;
uso = MEDICIONES.uso;
disparidad_mediana_px = nan(nMediciones, 1);
desviacion_disparidad_px = nan(nMediciones, 1);
frames_analizados = zeros(nMediciones, 1);
frames_validos = zeros(nMediciones, 1);
pixeles_validos_mediana = zeros(nMediciones, 1);

for i = 1:nMediciones
    fprintf('\n========================================\n');
    fprintf('PROCESANDO MEDICIÓN %d/%d\n', i, nMediciones);
    fprintf('========================================\n');
    fprintf('Hora: %s\n', string(MEDICIONES.timestamp(i)));
    fprintf('Altura manual: %.2f mm\n', MEDICIONES.altura_mm(i));

    resultado = procesarMedicion( ...
        MEDICIONES.timestamp(i), ...
        registro, ...
        videoL, ...
        videoR, ...
        stereoParams, ...
        tamCal, ...
        desplazamientoHorizontalR, ...
        ROI, ...
        minDisparity, ...
        maxDisparity, ...
        VENTANA_TEMPORAL_S, ...
        SALTO_FRAMES_VENTANA, ...
        GAMMA_CORRECCION, ...
        CLAHE_CLIP_LIMIT, ...
        CLAHE_NUM_TILES, ...
        BILATERAL_DEGREE, ...
        BILATERAL_SPATIAL_SIGMA, ...
        RADIO_DILATACION_BORDES, ...
        INTENSIDAD_MIN_VALIDA, ...
        INTENSIDAD_MAX_VALIDA, ...
        MIN_PIXELES_VALIDOS_FRAME, ...
        PERCENTIL_DISPARIDAD_ROI, ...
        IQR_FACTOR ...
    );

    timestamp_solicitado(i) = MEDICIONES.timestamp(i);
    timestamp_encontrado(i) = resultado.timestamp_encontrado;
    diferencia_temporal_s(i) = resultado.diferencia_temporal_s;
    frame_central(i) = resultado.frame_central;
    disparidad_mediana_px(i) = resultado.disparidad_mediana_px;
    desviacion_disparidad_px(i) = resultado.desviacion_disparidad_px;
    frames_analizados(i) = resultado.frames_analizados;
    frames_validos(i) = resultado.frames_validos;
    pixeles_validos_mediana(i) = resultado.pixeles_validos_mediana;

    fprintf('Frame central: %d\n', resultado.frame_central);
    fprintf('Frames válidos: %d/%d\n', ...
        resultado.frames_validos, resultado.frames_analizados);
    fprintf('Disparidad mediana: %.4f px\n', ...
        resultado.disparidad_mediana_px);
end

resultados = table( ...
    timestamp_solicitado, ...
    timestamp_encontrado, ...
    diferencia_temporal_s, ...
    frame_central, ...
    altura_mm, ...
    uso, ...
    disparidad_mediana_px, ...
    desviacion_disparidad_px, ...
    frames_analizados, ...
    frames_validos, ...
    pixeles_validos_mediana ...
);

rutaCSV = fullfile(CARPETA_SALIDA, ...
    'pares_disparidad_altura_matlab.csv');

writetable(resultados, rutaCSV);

fprintf('\nTabla guardada en:\n%s\n', rutaCSV);
disp(resultados);

%% ============================================================
% AJUSTAR MODELOS EMPÍRICOS
% =============================================================

[modelos, mejorModelo] = ajustarModelos(resultados);

fprintf('\n========================================\n');
fprintf('RESULTADOS DE LOS MODELOS\n');
fprintf('========================================\n');

nombresModelos = fieldnames(modelos);

for i = 1:numel(nombresModelos)
    nombre = nombresModelos{i};
    modelo = modelos.(nombre);

    fprintf('\nModelo: %s\n', nombre);
    fprintf('Coeficientes: ');
    fprintf('%.10g ', modelo.coeficientes);
    fprintf('\n');
    fprintf('RMSE calibración: %.4f mm\n', ...
        modelo.rmse_calibracion_mm);

    if isfinite(modelo.rmse_validacion_mm)
        fprintf('RMSE validación: %.4f mm\n', ...
            modelo.rmse_validacion_mm);
    end
end

fprintf('\nMejor modelo: %s\n', mejorModelo);

%% ============================================================
% GUARDAR MODELO Y GRÁFICO
% =============================================================

rutaMAT = fullfile(CARPETA_SALIDA, ...
    'modelo_altura_espuma_matlab.mat');

save( ...
    rutaMAT, ...
    'stereoParams', ...
    'ROI', ...
    'desplazamientoHorizontalR', ...
    'minDisparity', ...
    'maxDisparity', ...
    'modelos', ...
    'mejorModelo', ...
    'resultados' ...
);

rutaGrafico = fullfile(CARPETA_SALIDA, ...
    'grafico_calibracion_altura.png');

guardarGraficoModelos( ...
    resultados, ...
    modelos, ...
    mejorModelo, ...
    rutaGrafico ...
);

fprintf('\nModelo guardado en:\n%s\n', rutaMAT);
fprintf('\nGráfico guardado en:\n%s\n', rutaGrafico);

fprintf('\n========================================\n');
fprintf('PROCESAMIENTO FINALIZADO\n');
fprintf('========================================\n');

%% ============================================================
% FUNCIONES LOCALES
% =============================================================

function [frameLout, frameRout] = adaptarResolucionCalibracion( ...
    frameL, frameR, tamCal)

    if ~isequal(size(frameL, 1:2), tamCal)
        frameLout = imresize(frameL, tamCal);
    else
        frameLout = frameL;
    end

    if ~isequal(size(frameR, 1:2), tamCal)
        frameRout = imresize(frameR, tamCal);
    else
        frameRout = frameR;
    end
end


function verificarRectificacion(rectL, rectR)
    separacion = 60;

    montaje = imtile({rectL, rectR}, ...
        'GridSize', [1 2], ...
        'BackgroundColor', 'black');

    f = figure( ...
        'Name', 'Verificar rectificación', ...
        'NumberTitle', 'off', ...
        'Color', 'black' ...
    );

    imshow(montaje);
    hold on;

    alto = size(montaje, 1);

    for y = 1:separacion:alto
        yline(y, 'g-', 'LineWidth', 0.6);
    end

    title( ...
        ['Cámara 1 rectificada / Cámara 2 rectificada. ' ...
         'Los mismos detalles deben quedar en la misma fila.'], ...
        'Color', 'white' ...
    );

    respuesta = questdlg( ...
        '¿La rectificación vertical se ve correcta?', ...
        'Confirmar rectificación', ...
        'Sí, continuar', ...
        'No, detener', ...
        'Sí, continuar' ...
    );

    close(f);

    if ~strcmp(respuesta, 'Sí, continuar')
        error('Rectificación rechazada por el usuario.');
    end
end


function [desplazamientoR, d0] = seleccionarAlineacionHorizontal( ...
    rectL, rectR)

    f1 = figure( ...
        'Name', 'Punto en cámara 1', ...
        'NumberTitle', 'off' ...
    );

    imshow(rectL);
    title({ ...
        'CÁMARA 1: haz clic en un detalle reconocible.', ...
        'La coordenada vertical se mostrará, pero será ignorada.' ...
    });

    [xL, yL] = ginput(1);
    hold on;
    plot(xL, yL, 'r+', 'MarkerSize', 16, 'LineWidth', 2);
    drawnow;
    pause(0.4);
    close(f1);

    f2 = figure( ...
        'Name', 'Punto en cámara 2', ...
        'NumberTitle', 'off' ...
    );

    imshow(rectR);
    title({ ...
        'CÁMARA 2: haz clic en el detalle correspondiente.', ...
        'Solo se utilizará la diferencia horizontal.' ...
    });

    [xR, yR] = ginput(1);
    hold on;
    plot(xR, yR, 'r+', 'MarkerSize', 16, 'LineWidth', 2);
    drawnow;
    pause(0.4);
    close(f2);

    errorVertical = abs(yL - yR);

    fprintf('Selección manual:');

    fprintf('Cámara 1: x=%.3f, y=%.3f px', xL, yL);
    fprintf('Cámara 2: x=%.3f, y=%.3f px', xR, yR);
    fprintf('Diferencia vertical ignorada: %.3f px', errorVertical);

    % Únicamente se usa la separación horizontal.
    d0 = xL - xR;

    % Trasladar la imagen derecha para que xR + desplazamientoR = xL.
    desplazamientoR = d0;
end


function salida = desplazarHorizontal(imagen, desplazamientoX)
    salida = imtranslate( ...
        imagen, ...
        [desplazamientoX 0], ...
        'OutputView', 'same', ...
        'FillValues', 0 ...
    );
end


function ROI = seleccionarROI(imagen)
    f = figure( ...
        'Name', 'Seleccionar ROI de espuma', ...
        'NumberTitle', 'off' ...
    );

    imshow(imagen);
    title([ ...
        'Dibuja una ROI que contenga la espuma. ' ...
        'Evita la lanza y los bordes negros.' ...
    ]);

    h = drawrectangle('Color', 'yellow');

    if isempty(h)
        close(f);
        error('No se seleccionó una ROI.');
    end

    wait(h);
    ROI = round(h.Position);
    close(f);

    ROI(1) = max(1, ROI(1));
    ROI(2) = max(1, ROI(2));
    ROI(3) = max(1, ROI(3));
    ROI(4) = max(1, ROI(4));

    maxAncho = size(imagen, 2) - ROI(1) + 1;
    maxAlto = size(imagen, 1) - ROI(2) + 1;

    ROI(3) = min(ROI(3), maxAncho);
    ROI(4) = min(ROI(4), maxAlto);
end


function coincidencia = buscarFrameMasCercano(registro, timestampObjetivo)
    diferencias = abs(seconds(registro.timestamp - timestampObjetivo));
    [diferenciaMin, idx] = min(diferencias);

    frameGrabado = double(registro.frame_grabado(idx));

    % El CSV generado en Python normalmente usa índice base 0.
    frameMatlab = frameGrabado + 1;

    coincidencia.timestamp_encontrado = registro.timestamp(idx);
    coincidencia.diferencia_s = diferenciaMin;
    coincidencia.frame_grabado_csv = frameGrabado;
    coincidencia.frame_video_matlab = frameMatlab;
end


function frames = obtenerFramesVentana( ...
    registro, timestampCentral, ventanaS, salto)

    inicio = timestampCentral - seconds(ventanaS);
    fin = timestampCentral + seconds(ventanaS);

    filas = registro( ...
        registro.timestamp >= inicio & ...
        registro.timestamp <= fin, ...
        : ...
    );

    frames = double(filas.frame_grabado) + 1;

    if isempty(frames)
        return;
    end

    frames = frames(1:salto:end);
end


function resultado = procesarMedicion( ...
    timestampObjetivo, ...
    registro, ...
    videoL, ...
    videoR, ...
    stereoParams, ...
    tamCal, ...
    desplazamientoHorizontalR, ...
    ROI, ...
    minD, ...
    maxD, ...
    ventanaS, ...
    salto, ...
    gamma, ...
    clipLimit, ...
    numTiles, ...
    bilateralDegree, ...
    bilateralSigma, ...
    radioDilatacion, ...
    intensidadMin, ...
    intensidadMax, ...
    minPixeles, ...
    percentilDisp, ...
    iqrFactor)

    coincidencia = buscarFrameMasCercano( ...
        registro, timestampObjetivo ...
    );

    frames = obtenerFramesVentana( ...
        registro, timestampObjetivo, ventanaS, salto ...
    );

    listaDisp = [];
    listaPixeles = [];

    frameMax = floor(min( ...
        videoL.Duration * videoL.FrameRate, ...
        videoR.Duration * videoR.FrameRate ...
    ));

    for k = 1:numel(frames)
        idx = frames(k);

        if idx < 1 || idx > frameMax
            continue;
        end

        try
            frameL = read(videoL, idx);
            frameR = read(videoR, idx);
        catch
            continue;
        end

        [frameL, frameR] = adaptarResolucionCalibracion( ...
            frameL, frameR, tamCal ...
        );

        [rectL, rectR] = rectifyStereoImages( ...
            frameL, ...
            frameR, ...
            stereoParams, ...
            'OutputView', 'full' ...
        );

        rectR = desplazarHorizontal( ...
            rectR, ...
            desplazamientoHorizontalR ...
        );

        [dispMap, edgeMask, ~] = calcularDisparidadFrame( ...
            rectL, ...
            rectR, ...
            minD, ...
            maxD, ...
            gamma, ...
            clipLimit, ...
            numTiles, ...
            bilateralDegree, ...
            bilateralSigma, ...
            radioDilatacion, ...
            intensidadMin, ...
            intensidadMax ...
        );

        [dSuperficie, nPixeles] = obtenerDisparidadROI( ...
            dispMap, ...
            edgeMask, ...
            ROI, ...
            minPixeles, ...
            percentilDisp, ...
            iqrFactor ...
        );

        if isfinite(dSuperficie)
            listaDisp(end + 1, 1) = dSuperficie; %#ok<AGROW>
            listaPixeles(end + 1, 1) = nPixeles; %#ok<AGROW>
        end
    end

    if isempty(listaDisp)
        dMediana = NaN;
        dDesviacion = NaN;
        pixelesMediana = 0;
    else
        valores = filtroIQR(listaDisp, iqrFactor);

        if isempty(valores)
            valores = listaDisp;
        end

        dMediana = median(valores, 'omitnan');
        dDesviacion = std(valores, 'omitnan');
        pixelesMediana = round(median(listaPixeles, 'omitnan'));
    end

    resultado.timestamp_encontrado = ...
        coincidencia.timestamp_encontrado;

    resultado.diferencia_temporal_s = ...
        coincidencia.diferencia_s;

    resultado.frame_central = ...
        coincidencia.frame_video_matlab;

    resultado.disparidad_mediana_px = dMediana;
    resultado.desviacion_disparidad_px = dDesviacion;
    resultado.frames_analizados = numel(frames);
    resultado.frames_validos = numel(listaDisp);
    resultado.pixeles_validos_mediana = pixelesMediana;
end


function [dispMap, edgeMask, grayOriginalL] = ...
    calcularDisparidadFrame( ...
        rectL, ...
        rectR, ...
        minD, ...
        maxD, ...
        gamma, ...
        clipLimit, ...
        numTiles, ...
        bilateralDegree, ...
        bilateralSigma, ...
        radioDilatacion, ...
        intensidadMin, ...
        intensidadMax)

    grayOriginalL = im2gray(rectL);
    grayOriginalR = im2gray(rectR);

    grayL = im2uint8(imadjust( ...
        im2double(grayOriginalL), ...
        [], ...
        [], ...
        gamma ...
    ));

    grayR = im2uint8(imadjust( ...
        im2double(grayOriginalR), ...
        [], ...
        [], ...
        gamma ...
    ));

    grayL = adapthisteq( ...
        grayL, ...
        'ClipLimit', clipLimit, ...
        'NumTiles', numTiles ...
    );

    grayR = adapthisteq( ...
        grayR, ...
        'ClipLimit', clipLimit, ...
        'NumTiles', numTiles ...
    );

    grayL = imbilatfilt( ...
        grayL, ...
        bilateralDegree, ...
        bilateralSigma ...
    );

    grayR = imbilatfilt( ...
        grayR, ...
        bilateralDegree, ...
        bilateralSigma ...
    );

    edges = edge(grayL, 'Canny');

    se = strel('disk', radioDilatacion, 0);
    edgeMask = imdilate(edges, se);

    dispMap = disparitySGM( ...
        grayL, ...
        grayR, ...
        'DisparityRange', [minD maxD], ...
        'UniquenessThreshold', 10 ...
    );

    intensidadValida = ...
        grayOriginalL >= intensidadMin & ...
        grayOriginalL <= intensidadMax & ...
        grayOriginalR >= intensidadMin & ...
        grayOriginalR <= intensidadMax;

    invalidos = ...
        ~isfinite(dispMap) | ...
        dispMap <= minD + 2 | ...
        dispMap >= maxD - 2 | ...
        ~intensidadValida;

    dispMap(invalidos) = NaN;
end


function [dSuperficie, cantidad] = obtenerDisparidadROI( ...
    dispMap, edgeMask, ROI, minPixeles, percentilDisp, iqrFactor)

    x = ROI(1);
    y = ROI(2);
    w = ROI(3);
    h = ROI(4);

    x2 = min(size(dispMap, 2), x + w - 1);
    y2 = min(size(dispMap, 1), y + h - 1);

    roiDisp = dispMap(y:y2, x:x2);
    roiEdges = edgeMask(y:y2, x:x2);

    validos = isfinite(roiDisp) & roiEdges;

    valores = roiDisp(validos);
    cantidad = numel(valores);

    if cantidad < minPixeles
        dSuperficie = NaN;
        return;
    end

    valores = filtroIQR(valores, iqrFactor);

    if numel(valores) < minPixeles
        dSuperficie = NaN;
        cantidad = numel(valores);
        return;
    end

    dSuperficie = prctile(valores, percentilDisp);
    cantidad = numel(valores);
end


function valoresFiltrados = filtroIQR(valores, factor)
    valores = valores(isfinite(valores));

    if isempty(valores)
        valoresFiltrados = valores;
        return;
    end

    q1 = prctile(valores, 25);
    q3 = prctile(valores, 75);
    iqrValor = q3 - q1;

    if iqrValor <= 0
        valoresFiltrados = valores;
        return;
    end

    limiteBajo = q1 - factor * iqrValor;
    limiteAlto = q3 + factor * iqrValor;

    valoresFiltrados = valores( ...
        valores >= limiteBajo & ...
        valores <= limiteAlto ...
    );
end


function mostrarDiagnosticoDisparidad( ...
    rectL, dispMap, edgeMask, grayL, ROI, minD, maxD)

    f = figure( ...
        'Name', 'Diagnóstico de disparidad', ...
        'NumberTitle', 'off' ...
    );

    tiledlayout(2, 2);

    nexttile;
    imshow(rectL);
    hold on;
    rectangle( ...
        'Position', ROI, ...
        'EdgeColor', 'yellow', ...
        'LineWidth', 2 ...
    );
    title('Cámara izquierda rectificada + ROI');

    nexttile;
    imagesc(dispMap, [minD maxD]);
    axis image off;
    colorbar;
    hold on;
    rectangle( ...
        'Position', ROI, ...
        'EdgeColor', 'white', ...
        'LineWidth', 2 ...
    );
    title('Mapa de disparidad');

    nexttile;
    imshow(edgeMask);
    hold on;
    rectangle( ...
        'Position', ROI, ...
        'EdgeColor', 'yellow', ...
        'LineWidth', 2 ...
    );
    title('Máscara de bordes');

    nexttile;
    imshow(grayL);
    hold on;
    rectangle( ...
        'Position', ROI, ...
        'EdgeColor', 'yellow', ...
        'LineWidth', 2 ...
    );
    title('Escala de grises original');

    drawnow;
end


function [modelos, mejorNombre] = ajustarModelos(resultados)
    calibracion = resultados( ...
        resultados.uso == "calibracion" & ...
        isfinite(resultados.disparidad_mediana_px), ...
        : ...
    );

    validacion = resultados( ...
        resultados.uso == "validacion" & ...
        isfinite(resultados.disparidad_mediana_px), ...
        : ...
    );

    if height(calibracion) < 3
        error(['Se necesitan al menos tres mediciones válidas ' ...
               'de calibración.']);
    end

    xCal = calibracion.disparidad_mediana_px;
    yCal = calibracion.altura_mm;

    if range(xCal) < 0.5
        error(['Las disparidades casi no cambian. ' ...
               'No se ajustará un modelo falso.']);
    end

    modelos = struct();

    modelos.lineal.tipo = 'polinomio';
    modelos.lineal.grado = 1;
    modelos.lineal.coeficientes = polyfit(xCal, yCal, 1);

    if height(calibracion) >= 4
        modelos.cuadratico.tipo = 'polinomio';
        modelos.cuadratico.grado = 2;
        modelos.cuadratico.coeficientes = polyfit(xCal, yCal, 2);
    end

    if all(abs(xCal) > 1e-9)
        modelos.inverso.tipo = 'inverso';
        modelos.inverso.grado = 1;
        modelos.inverso.coeficientes = polyfit(1 ./ xCal, yCal, 1);
    end

    nombres = fieldnames(modelos);

    for k = 1:numel(nombres)
        nombre = nombres{k};
        modelo = modelos.(nombre);

        predCal = predecirModelo(modelo, xCal);

        modelos.(nombre).mae_calibracion_mm = ...
            mean(abs(predCal - yCal));

        modelos.(nombre).rmse_calibracion_mm = ...
            sqrt(mean((predCal - yCal).^2));

        if ~isempty(validacion)
            xVal = validacion.disparidad_mediana_px;
            yVal = validacion.altura_mm;
            predVal = predecirModelo(modelo, xVal);

            modelos.(nombre).mae_validacion_mm = ...
                mean(abs(predVal - yVal));

            modelos.(nombre).rmse_validacion_mm = ...
                sqrt(mean((predVal - yVal).^2));
        else
            modelos.(nombre).mae_validacion_mm = NaN;
            modelos.(nombre).rmse_validacion_mm = NaN;
        end
    end

    puntajes = nan(numel(nombres), 1);

    for k = 1:numel(nombres)
        if ~isempty(validacion)
            puntajes(k) = modelos.(nombres{k}).rmse_validacion_mm;
        else
            puntajes(k) = modelos.(nombres{k}).rmse_calibracion_mm;
        end
    end

    [~, idxMejor] = min(puntajes);
    mejorNombre = nombres{idxMejor};
end


function y = predecirModelo(modelo, x)
    switch modelo.tipo
        case 'polinomio'
            y = polyval(modelo.coeficientes, x);

        case 'inverso'
            y = polyval(modelo.coeficientes, 1 ./ x);

        otherwise
            error('Tipo de modelo desconocido.');
    end
end


function guardarGraficoModelos( ...
    resultados, modelos, mejorNombre, ruta)

    validos = resultados( ...
        isfinite(resultados.disparidad_mediana_px), ...
        : ...
    );

    if isempty(validos)
        return;
    end

    calibracion = validos(validos.uso == "calibracion", :);
    validacion = validos(validos.uso == "validacion", :);

    xMin = min(validos.disparidad_mediana_px);
    xMax = max(validos.disparidad_mediana_px);
    xCurva = linspace(xMin, xMax, 300);

    modelo = modelos.(mejorNombre);
    yCurva = predecirModelo(modelo, xCurva);

    f = figure('Visible', 'off');

    scatter( ...
        calibracion.disparidad_mediana_px, ...
        calibracion.altura_mm, ...
        70, ...
        'filled' ...
    );

    hold on;

    if ~isempty(validacion)
        scatter( ...
            validacion.disparidad_mediana_px, ...
            validacion.altura_mm, ...
            70, ...
            'filled' ...
        );
    end

    plot(xCurva, yCurva, 'LineWidth', 2);

    xlabel('Disparidad representativa [px]');
    ylabel('Altura conocida [mm]');
    title(sprintf( ...
        'Calibración empírica disparidad-altura (%s)', ...
        mejorNombre ...
    ));

    grid on;

    if isempty(validacion)
        legend('Calibración', 'Modelo', 'Location', 'best');
    else
        legend( ...
            'Calibración', ...
            'Validación', ...
            'Modelo', ...
            'Location', ...
            'best' ...
        );
    end

    exportgraphics(f, ruta, 'Resolution', 300);
    close(f);
end
