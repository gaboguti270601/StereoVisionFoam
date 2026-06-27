%% reproducir_canny_video_matlab.m
% Conversión a MATLAB del código Python/OpenCV proporcionado.
%
% Requisitos:
%   - Image Processing Toolbox
%
% Controles:
%   Espacio : pausar/reanudar
%   R       : reiniciar desde START_FRAME
%   Q o ESC : salir

clear;
clc;
close all;

%% =============================================================
% CONFIGURACIÓN
% =============================================================

VIDEO_PATH = "D:\MDT\Stereovision\II38P_cam1.mp4";
START_FRAME = 19000;   % Numeración desde 0, igual que en OpenCV

% -------------------------------------------------------------
% CLAHE
% -------------------------------------------------------------
CLAHE_CLIP_LIMIT_OPENCV = 1.2;
CLAHE_GRID_SIZE = [8 8];

% Solo modifica la visualización de la imagen CLAHE.
CLAHE_BRIGHTNESS_ALPHA = 0.45;
CLAHE_BRIGHTNESS_BETA = 0;

% -------------------------------------------------------------
% Filtro bilateral
% -------------------------------------------------------------
BILATERAL_DIAMETER = 9; %#ok<NASGU>
BILATERAL_SIGMA_COLOR = 45;
BILATERAL_SIGMA_SPACE = 45;

% -------------------------------------------------------------
% Canny automático por gradiente
% -------------------------------------------------------------
LOW_PERCENTILE = 70;
HIGH_PERCENTILE = 90;

% MATLAB no permite seleccionar directamente apertureSize=3, 5 o 7
% en edge(...,"Canny"). Se conserva la variable como referencia.
CANNY_APERTURE_SIZE = 3;

% -------------------------------------------------------------
% Cierre morfológico
% -------------------------------------------------------------
USE_MORPH_CLOSE = false;
MORPH_KERNEL_SIZE = 3;

%% =============================================================
% EJECUCIÓN
% =============================================================

reproduceCannyVideo( ...
    VIDEO_PATH, ...
    START_FRAME, ...
    CLAHE_CLIP_LIMIT_OPENCV, ...
    CLAHE_GRID_SIZE, ...
    CLAHE_BRIGHTNESS_ALPHA, ...
    CLAHE_BRIGHTNESS_BETA, ...
    BILATERAL_SIGMA_COLOR, ...
    BILATERAL_SIGMA_SPACE, ...
    LOW_PERCENTILE, ...
    HIGH_PERCENTILE, ...
    CANNY_APERTURE_SIZE, ...
    USE_MORPH_CLOSE, ...
    MORPH_KERNEL_SIZE);


%% =============================================================
% FUNCIONES LOCALES
% =============================================================

function reproduceCannyVideo( ...
    videoPath, ...
    startFrame, ...
    claheClipLimitOpenCV, ...
    claheGridSize, ...
    claheBrightnessAlpha, ...
    claheBrightnessBeta, ...
    bilateralSigmaColor, ...
    bilateralSigmaSpace, ...
    lowPercentile, ...
    highPercentile, ...
    cannyApertureSize, ...
    useMorphClose, ...
    morphKernelSize)

    if ~isfile(videoPath)
        error("No se encontró el video: %s", videoPath);
    end

    video = VideoReader(videoPath);

    fps = video.FrameRate;
    if isempty(fps) || fps <= 0
        fps = 30;
    end

    totalFrames = floor(video.Duration * fps);

    if startFrame < 0 || startFrame >= totalFrames
        error( ...
            "START_FRAME=%d está fuera del rango válido [0, %d].", ...
            startFrame, totalFrames - 1);
    end

    if lowPercentile < 0
        lowPercentile = 0;
    end

    if highPercentile > 100
        highPercentile = 100;
    end

    if lowPercentile >= highPercentile
        error("LOW_PERCENTILE debe ser menor que HIGH_PERCENTILE.");
    end

    if ~ismember(cannyApertureSize, [3 5 7])
        cannyApertureSize = 3;
    end

    if morphKernelSize < 1
        morphKernelSize = 1;
    end

    if mod(morphKernelSize, 2) == 0
        morphKernelSize = morphKernelSize + 1;
    end

    % OpenCV y MATLAB utilizan definiciones diferentes de ClipLimit.
    % Esta conversión conserva aproximadamente el ajuste 1.2 de OpenCV.
    matlabClaheClipLimit = min(max(claheClipLimitOpenCV / 100, 0), 1);

    video.CurrentTime = startFrame / fps;

    firstFrame = readFrame(video);
    [height, width, ~] = size(firstFrame);

    video.CurrentTime = startFrame / fps;

    scale = getDisplayScale(width, height, 0.45);
    displayWidth = max(1, round(width * scale));
    displayHeight = max(1, round(height * scale));

    fprintf("--------------------------------------\n");
    fprintf("Información del video\n");
    fprintf("--------------------------------------\n");
    fprintf("Resolución: %d x %d\n", width, height);
    fprintf("FPS: %.2f\n", fps);
    fprintf("Total aproximado de frames: %d\n", totalFrames);
    fprintf("Frame inicial: %d\n", startFrame);

    fprintf("\nParámetros CLAHE:\n");
    fprintf("Clip limit OpenCV: %.3f\n", claheClipLimitOpenCV);
    fprintf("Clip limit MATLAB aproximado: %.4f\n", matlabClaheClipLimit);
    fprintf("Grid size: [%d %d]\n", claheGridSize(1), claheGridSize(2));
    fprintf("Alpha visual CLAHE: %.3f\n", claheBrightnessAlpha);

    fprintf("\nFiltro bilateral:\n");
    fprintf("Sigma color: %.3f\n", bilateralSigmaColor);
    fprintf("Sigma espacio: %.3f\n", bilateralSigmaSpace);

    fprintf("\nCanny automático:\n");
    fprintf("Percentil inferior: %.1f\n", lowPercentile);
    fprintf("Percentil superior: %.1f\n", highPercentile);
    fprintf("Aperture size de referencia: %d\n", cannyApertureSize);

    fprintf("\nMorfología:\n");
    fprintf("Cierre activado: %d\n", useMorphClose);
    fprintf("Kernel morfológico: %d\n", morphKernelSize);

    fprintf("\nControles:\n");
    fprintf("Espacio: pausar o reanudar\n");
    fprintf("R: reiniciar\n");
    fprintf("Q o ESC: salir\n");

    % Estado compartido entre las ventanas.
    state.paused = false;
    state.stop = false;
    state.restart = false;

    figOriginal = figure( ...
        "Name", "Video original", ...
        "NumberTitle", "off", ...
        "MenuBar", "none", ...
        "ToolBar", "none", ...
        "KeyPressFcn", @keyCallback, ...
        "CloseRequestFcn", @closeCallback);

    figClahe = figure( ...
        "Name", sprintf( ...
            "Video CLAHE oscurecido - alpha=%.2f", ...
            claheBrightnessAlpha), ...
        "NumberTitle", "off", ...
        "MenuBar", "none", ...
        "ToolBar", "none", ...
        "KeyPressFcn", @keyCallback, ...
        "CloseRequestFcn", @closeCallback);

    figEdges = figure( ...
        "Name", "Video Canny automático por gradiente", ...
        "NumberTitle", "off", ...
        "MenuBar", "none", ...
        "ToolBar", "none", ...
        "KeyPressFcn", @keyCallback, ...
        "CloseRequestFcn", @closeCallback);

    setappdata(figOriginal, "state", state);
    setappdata(figClahe, "state", state);
    setappdata(figEdges, "state", state);

    axOriginal = axes(figOriginal);
    axClahe = axes(figClahe);
    axEdges = axes(figEdges);

    hOriginal = imshow(zeros(displayHeight, displayWidth, 3, "uint8"), ...
        "Parent", axOriginal);
    hClahe = imshow(zeros(displayHeight, displayWidth, "uint8"), ...
        "Parent", axClahe);
    hEdges = imshow(false(displayHeight, displayWidth), ...
        "Parent", axEdges);

    currentFrame = startFrame;
    frameDelay = 1 / fps;

    while true
        state = getSharedState(figOriginal, figClahe, figEdges);

        if state.stop
            break;
        end

        if state.restart
            video.CurrentTime = startFrame / fps;
            currentFrame = startFrame;

            state.restart = false;
            state.paused = false;
            setSharedState(state, figOriginal, figClahe, figEdges);

            fprintf("Video reiniciado desde el frame %d.\n", startFrame);
        end

        if state.paused
            drawnow;
            pause(0.03);
            continue;
        end

        if ~hasFrame(video)
            fprintf("Fin del video.\n");
            break;
        end

        frameStart = tic;

        frame = readFrame(video);

        % 1. Escala de grises
        if size(frame, 3) == 3
            gray = im2gray(frame);
        else
            gray = frame;
        end

        gray = im2uint8(gray);

        % 2. CLAHE
        grayClahe = adapthisteq( ...
            gray, ...
            "NumTiles", claheGridSize, ...
            "ClipLimit", matlabClaheClipLimit, ...
            "Distribution", "uniform");

        % 3. Oscurecer CLAHE solo para visualización
        grayClaheDisplay = uint8(min(max( ...
            double(grayClahe) .* claheBrightnessAlpha + ...
            claheBrightnessBeta, ...
            0), 255));

        % 4. Filtro bilateral
        % DegreeOfSmoothing se relaciona con sigmaColor^2.
        degreeOfSmoothing = bilateralSigmaColor^2;

        grayFiltered = imbilatfilt( ...
            grayClahe, ...
            degreeOfSmoothing, ...
            bilateralSigmaSpace);

        % 5. Canny automático basado en percentiles del gradiente
        [edges, autoLow, autoHigh] = automaticCannyFromGradient( ...
            grayFiltered, ...
            lowPercentile, ...
            highPercentile);

        % 6. Cierre morfológico opcional
        if useMorphClose
            se = strel("disk", floor(morphKernelSize / 2), 0);
            edges = imclose(edges, se);
        end

        % 7. Redimensionar para mostrar
        frameSmall = imresize( ...
            frame, ...
            [displayHeight displayWidth], ...
            "bilinear");

        claheSmall = imresize( ...
            grayClaheDisplay, ...
            [displayHeight displayWidth], ...
            "bilinear");

        edgesSmall = imresize( ...
            edges, ...
            [displayHeight displayWidth], ...
            "nearest");

        % Mostrar resultados
        if isgraphics(hOriginal)
            hOriginal.CData = frameSmall;
            title(axOriginal, sprintf("Frame: %d", currentFrame), ...
                "Color", "white");
        end

        if isgraphics(hClahe)
            hClahe.CData = claheSmall;
            title(axClahe, sprintf("Frame: %d", currentFrame), ...
                "Color", "white");
        end

        if isgraphics(hEdges)
            hEdges.CData = edgesSmall;
            title(axEdges, sprintf( ...
                "Frame: %d | Canny: %d-%d", ...
                currentFrame, autoLow, autoHigh), ...
                "Color", "white");
        end

        drawnow limitrate;

        currentFrame = currentFrame + 1;

        elapsed = toc(frameStart);
        if elapsed < frameDelay
            pause(frameDelay - elapsed);
        end
    end

    figures = [figOriginal figClahe figEdges];
    figures = figures(isgraphics(figures));
    delete(figures);


    function keyCallback(~, event)
        stateNow = getSharedState(figOriginal, figClahe, figEdges);

        switch lower(event.Key)
            case "space"
                stateNow.paused = ~stateNow.paused;

                if stateNow.paused
                    fprintf("Video pausado.\n");
                else
                    fprintf("Video reanudado.\n");
                end

            case "r"
                stateNow.restart = true;

            case {"q", "escape"}
                stateNow.stop = true;
        end

        setSharedState(stateNow, figOriginal, figClahe, figEdges);
    end


    function closeCallback(~, ~)
        stateNow = getSharedState(figOriginal, figClahe, figEdges);
        stateNow.stop = true;
        setSharedState(stateNow, figOriginal, figClahe, figEdges);
    end
end


function [edges, thresholdLow, thresholdHigh] = ...
    automaticCannyFromGradient(grayImage, lowPercentile, highPercentile)

    graySingle = single(grayImage);

    % Gradientes Sobel horizontal y vertical.
    [sobelX, sobelY] = imgradientxy(graySingle, "sobel");

    magnitude = hypot(sobelX, sobelY);
    validGradients = magnitude(magnitude > 0);

    if isempty(validGradients)
        edges = false(size(grayImage));
        thresholdLow = 0;
        thresholdHigh = 1;
        return;
    end

    thresholdLow = floor(prctile(validGradients, lowPercentile));
    thresholdHigh = floor(prctile(validGradients, highPercentile));

    thresholdLow = max(1, min(thresholdLow, 254));
    thresholdHigh = max(thresholdLow + 1, min(thresholdHigh, 255));

    % edge(...,"Canny") recibe umbrales normalizados entre 0 y 1.
    normalizedLow = thresholdLow / 255;
    normalizedHigh = thresholdHigh / 255;

    normalizedLow = min(max(normalizedLow, eps), 1 - eps);
    normalizedHigh = min(max(normalizedHigh, ...
        normalizedLow + eps), 1);

    edges = edge( ...
        grayImage, ...
        "Canny", ...
        [normalizedLow normalizedHigh]);
end


function scale = getDisplayScale(width, height, windowFraction)
    screenSize = get(groot, "ScreenSize");

    screenWidth = screenSize(3);
    screenHeight = screenSize(4);

    scaleWidth = (screenWidth * windowFraction) / width;
    scaleHeight = (screenHeight * 0.85) / height;

    scale = min([scaleWidth, scaleHeight, 1]);
end


function state = getSharedState(varargin)
    state = struct( ...
        "paused", false, ...
        "stop", false, ...
        "restart", false);

    for k = 1:nargin
        fig = varargin{k};

        if isgraphics(fig) && isappdata(fig, "state")
            candidate = getappdata(fig, "state");

            state.paused = state.paused || candidate.paused;
            state.stop = state.stop || candidate.stop;
            state.restart = state.restart || candidate.restart;
        end
    end
end


function setSharedState(state, varargin)
    for k = 1:numel(varargin)
        fig = varargin{k};

        if isgraphics(fig)
            setappdata(fig, "state", state);
        end
    end
end
