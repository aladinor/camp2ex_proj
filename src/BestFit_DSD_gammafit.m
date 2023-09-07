% This program computes fit resitual as a function of (N0, mu, lambda)
clear all; clc; fs = 18; close all; 
set(0,'defaultaxesfontsize',fs);
set(0,'defaulttextfontsize',fs);

%%% specify ranges of (N0, mu, lambda)
N0range = 10.^linspace(-5,10,100);  % originally used (-5,10,100)
murange = linspace(-2,15,101);  % originally used (-0.9,9.1,101)
lambdarange = linspace(5,505,102);

%%**Read in Files
fid = fopen('/data/snesbitt/h/gleiche1/LPVEx/2DVD/EMASALO/09_21/09_21_Emasalo_dsd.txt');   %DSD calculated using observed fall speed
C = textscan(fid, '%f %f %f %f %f %f %f %f %f');
fclose(fid);
start_hour    = C{1};
start_min     = C{2};
n_D_e         = C{9};   %in drops/m^3mm

%convert n_D to cm^-3 um^-1
n_D = n_D_e.*((1/100)^3).*(1/1000);

   data = ones(235,50)*NaN; % Allocate memory
 hour_e = ones(235,50)*NaN;
  min_e = ones(235,50)*NaN;
 
    count = 0;
for c = 1:50:length(start_hour)
    count = count+1;
    data(count,:) = n_D(c:(c+49));
   hour_e(count,:) = start_hour(c:(c+49));
    min_e(count,:) = start_min(c:(c+49));
end

%Create Time array
    tsec = ones(length(count))*NaN;
for it = 1:count
    tsec(it) = (hour_e(it,1)*3600)+(min_e(it,1)*60);
end

 bins_end = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, ...
             3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, ...
             6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, ...
             9.0, 9.2, 9.4, 9.6, 9.8, 10.0];
         
%Calculate bin midpoints and convert to cm
for i = 1:length(bins_end)-1
    bins_mid(i,1) = (bins_end(i+1) + bins_end(i))/2;
    Bins(i) = bins_mid(i,1)/(10);
end

%calculate difference in bins
   bins_diff_ori(1) = Bins(1) - 0;
for i = 2:length(Bins)
   bins_diff_ori(1,i) = Bins(i) - Bins(i-1);   
end
for i = 1:length(Bins)
   bins_ori(1,i) = Bins(i); 
end

bins = bins_ori;
bins_diff = bins_diff_ori;

fit_moments = [0 1 2];

for ii = 1:count
    sd = data(ii,:);
    sd(isnan(sd) == 1) = 0;
    sd(isinf(sd) == 1) = 0;
    
    M = zeros(7,1); % initialize M
    for i = 1:7
        M(i) = sum(sd .* bins.^(i-1) .* bins_diff); % M_rect
    end
    
    nonzeroidx = find(sd ~= 0); % eliminate zero terms, or normalized gamma will return inf
    sd = sd(nonzeroidx); 
    
    options_nonlin = optimset('Display','none','tolfun',1e-16,'tolx',1e-10,...
        'MaxFunEvals',3000,'MaxIter',1000,'Largescale','off','Algorithm','levenberg-marquardt');
        
    [estimates,~,~,exitflag,~] = lsqnonlin(@gamma_moment_inc_fit_nonlin,[1 1 100],...
        [0,-1,0]+1e-99,[1e20,100,1000],options_nonlin,bins,bins_diff,M(fit_moments+1),fit_moments);

    N0(ii,1) = estimates(1);
    mu(ii,1) = estimates(2);
    lambda(ii,1) = estimates(3);
    
    N_D(ii,:) = N0(ii,1).*bins_ori.^mu(ii,1).*exp(-lambda(ii,1).*bins_ori);
   
end


 
% hold on
% plot(bins_ori,N_D,'r')
% stairs(bins_ori,mean(data(:,:)))

save('BestFit_DSD_gammafit_Emasalo_09_21','N0','mu','lambda','N_D','tsec')
 
