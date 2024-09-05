% Read the data from the CSV file
data = readtable('Resultados/results.csv');

vars = ["dt", "t1", "s_t0", "Npoints"];

vars_analysis = nchoosek(vars,2);

fix.dt= 4;
fix.t1 = 4;
fix.s_t0 = 0.1;
fix.Npoints = 3;

for k_an = 1:size(vars_analysis,1)
    % Check which variables are not in the analysis (fixed)
    if ~any(strcmp('dt',vars_analysis(k_an,:)))
        an.dt = fix.dt;
    else
        an.dt = unique(data.dt);
    end

    if ~any(strcmp('t1',vars_analysis(k_an,:)))
        an.t1 = fix.t1;
    else
        an.t1 = unique(data.t1);
    end

    if ~any(strcmp('s_t0',vars_analysis(k_an,:)))
        an.s_t0 = fix.s_t0;
    else
        an.s_t0 = unique(data.s_t0);
    end

    if ~any(strcmp('Npoints',vars_analysis(k_an,:)))
        an.Npoints = fix.Npoints;
    else
        an.Npoints = unique(data.Npoints);
    end

    varX = vars_analysis{k_an,1};
    varY = vars_analysis{k_an,2};

    xVal = unique(data.(varX));
    yVal = unique(data.(varY));

    xVal = xVal(~isnan(xVal));
    yVal = yVal(~isnan(yVal));

    z_rmse = nan(length(xVal), length(yVal));
    z_r2 = nan(length(xVal), length(yVal));
    z_t2 = nan(length(xVal), length(yVal));
    z_s_r2 = nan(length(xVal), length(yVal));
    z_s_t2 = nan(length(xVal), length(yVal));

    ind = ismember(data.dt, an.dt) & ismember(data.t1, an.t1) & ismember(data.s_t0, an.s_t0) & ismember(data.Npoints, an.Npoints);
    data_select = data(ind,:);

    for kTab = 1:height(data_select)
        x = data_select.(varX)(kTab);
        y = data_select.(varY)(kTab);

        i = find(xVal == x);
        j = find(yVal == y);

        z_rmse(i,j) = sqrt(data_select.mean_SSE(kTab)\data_select.Npoints(kTab));
        z_r2(i,j) = data_select.mean_Resistance(kTab);
        z_t2(i,j) = data_select.mean_Temperature(kTab);
        z_s_r2(i,j) = data_select.std_Resistance(kTab);
        z_s_t2(i,j) = data_select.std_Temperature(kTab);
    end

    % Create a figure for the heatmaps
    figure;

    % Plot R2 heatmap
    subplot(2, 2, 1);
    heatmap(xVal, yVal, z_r2');
    title('R2 Heatmap');
    xlabel(varX);
    ylabel(varY);

    % Plot T2 heatmap
    subplot(2, 2, 2);
    h1 = heatmap(xVal, yVal, z_t2');
    title('T2 Heatmap');
    xlabel(varX);
    ylabel(varY);

    % Plot sR2 heatmap
    subplot(2, 2, 3);
    if any(strcmp('s_t0', vars_analysis(k_an,:)))
        z_s_r2 = log10(z_s_r2);
    end
    h2 = heatmap(xVal, yVal, z_s_r2');
    title('sR2 Heatmap');
    xlabel(varX);
    ylabel(varY);
    colorbar;

    % Plot sT2 heatmap
    subplot(2, 2, 4);
    if any(strcmp('s_t0', vars_analysis(k_an,:)))
        z_s_t2 = log10(z_s_t2);
    end
    h3 = heatmap(xVal, yVal, z_s_t2');
    title('sT2 Heatmap');
    xlabel(varX);
    ylabel(varY);
    colorbar;
    

    % Adjust layout
    sgtitle('Heatmap Plots');

end