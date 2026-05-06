% ================================================================
%  generate_MU_dataset.m
%
%  Self-contained script -- SimRIS_v18 is embedded below.
%  No path setup needed. Run from ANY folder.
%
%  Generates: RIS_Channels_MU_K4.mat
%    H   : (64, 16, 48000)  BS -> RIS       (complex)
%    G_1 : ( 1, 64, 48000)  RIS -> UE1  (45,45,1)
%    G_2 : ( 1, 64, 48000)  RIS -> UE2  (38,43,1)
%    G_3 : ( 1, 64, 48000)  RIS -> UE3  (44,42,1)
%    G_4 : ( 1, 64, 48000)  RIS -> UE4  (36,47,1)
%    D   : ( 1, 16, 48000)  Direct BS->UE1  (complex)
% ================================================================
clear; clc;

%% ---- Parameters ----
Environment = 1;   % Indoor InH Office
Scenario    = 1;   % xz plane (Scenario 1)
Frequency   = 28;  % GHz
ArrayType   = 1;   % ULA
N    = 64;
Nt   = 16;
Nr   = 1;
Nsym = 48000;

Tx_xyz  = [0,  25, 2];
RIS_xyz = [40, 50, 2];

Rx_all = [45, 45, 1;   % UE1  d_RIS=7.1m
          38, 43, 1;   % UE2  d_RIS=7.4m
          44, 42, 1;   % UE3  d_RIS=9.0m
          36, 47, 1];  % UE4  d_RIS=5.1m
K = size(Rx_all, 1);

%% ---- Preallocate ----
H   = complex(zeros(N,  Nt, Nsym));
D   = complex(zeros(Nr, Nt, Nsym));
G_1 = complex(zeros(Nr, N, Nsym));
G_2 = complex(zeros(Nr, N, Nsym));
G_3 = complex(zeros(Nr, N, Nsym));
G_4 = complex(zeros(Nr, N, Nsym));
G_all = {G_1, G_2, G_3, G_4};

%% ---- Generate ----
for k = 1:K
    Rx_k = Rx_all(k,:);
    fprintf('\nRun %d/%d  UE%d at [%g,%g,%g] ...\n', k, K, k, Rx_k);
    tic
    for s = 1:Nsym
        [h_s, g_s, d_s] = SimRIS_v18(Environment, Scenario, Frequency, ...
                                       ArrayType, N, Nt, Nr, ...
                                       Tx_xyz, Rx_k, RIS_xyz);
        if k == 1
            H(:,:,s) = h_s;
            D(:,:,s) = d_s;
        end
        G_all{k}(:,:,s) = transpose(g_s);   % (Nr x N) same as GUI

        if mod(s, 5000) == 0
            elapsed = toc;
            eta = elapsed / s * (Nsym - s);
            fprintf('  %5d/%d  elapsed=%.1fs  ETA=%.1fs\n', s, Nsym, elapsed, eta);
        end
    end
    fprintf('  UE%d done in %.1f sec\n', k, toc);
end

%% ---- Save ----
G_1 = G_all{1};  G_2 = G_all{2};
G_3 = G_all{3};  G_4 = G_all{4};

save_path = 'RIS_Channels_MU_K4.mat';
save(save_path, 'H', 'G_1', 'G_2', 'G_3', 'G_4', 'D', '-v7.3');
fprintf('\n=== DONE ===\n');
fprintf('Saved: %s\n', save_path);
fprintf('H:%s  G_1:%s  G_2:%s  G_3:%s  G_4:%s\n', ...
    mat2str(size(H)), mat2str(size(G_1)), mat2str(size(G_2)), ...
    mat2str(size(G_3)), mat2str(size(G_4)));


% ================================================================
% SimRIS_v18 -- embedded from SimRIS_GUI.m (CoreLab, Koc University)
% ================================================================
function [h,g,h_SISO]=SimRIS_v18(Environment,Scenario,Frequency,ArrayType,N,Nt,Nr,Tx_xyz,Rx_xyz,RIS_xyz)

lambda=(3*10^8)/(Frequency*10^9);
k=2*pi/lambda;
dis=lambda/2;

if mod(sqrt(N),1)~=0
    error('N should be an integer power of 2')
end

if Environment==1
    n_NLOS=3.19; sigma_NLOS=8.29; b_NLOS=0.06; f0=24.2;
    n_LOS=1.73;  sigma_LOS=3.02;  b_LOS=0;
else
    n_NLOS=3.19; sigma_NLOS=8.2;  b_NLOS=0;    f0=24.2;
    n_LOS=1.98;  sigma_LOS=3.1;   b_LOS=0;
end

if Frequency==28,     lambda_p=1.8;
elseif Frequency==73, lambda_p=1.9; end

q=0.285; Gain=pi;

x_Tx=Tx_xyz(1);y_Tx=Tx_xyz(2);z_Tx=Tx_xyz(3);
x_Rx=Rx_xyz(1);y_Rx=Rx_xyz(2);z_Rx=Rx_xyz(3);
x_RIS=RIS_xyz(1);y_RIS=RIS_xyz(2);z_RIS=RIS_xyz(3);

d_T_RIS = norm(Tx_xyz-RIS_xyz);

if Environment==1
    if z_RIS<z_Tx
        if d_T_RIS<=1.2,          p_LOS=1;
        elseif d_T_RIS<6.5,       p_LOS=exp(-(d_T_RIS-1.2)/4.7);
        else,                      p_LOS=0.32*exp(-(d_T_RIS-6.5)/32.6); end
        I_LOS=randsrc(1,1,[1,0;p_LOS 1-p_LOS]);
    else
        I_LOS=1;
    end
else
    p_LOS=min([20/d_T_RIS,1])*(1-exp(-d_T_RIS/39))+exp(-d_T_RIS/39);
    I_LOS=randsrc(1,1,[1,0;p_LOS 1-p_LOS]);
end

if I_LOS==1
    if Scenario==1
        I_phi=sign(x_RIS-x_Tx);
        phi_T_RIS_LOS=I_phi*atand(abs(x_RIS-x_Tx)/abs(y_RIS-y_Tx));
        I_theta=sign(z_Tx-z_RIS);
        theta_T_RIS_LOS=I_theta*asind(abs(z_RIS-z_Tx)/d_T_RIS);
        I_phi_Tx=sign(y_Tx-y_RIS);
        phi_Tx_LOS=I_phi_Tx*atand(abs(y_Tx-y_RIS)/abs(x_Tx-x_RIS));
        I_theta_Tx=sign(z_Tx-z_RIS);
        theta_Tx_LOS=I_theta_Tx*asind(abs(z_RIS-z_Tx)/d_T_RIS);
    else
        I_phi=sign(y_Tx-y_RIS);
        phi_T_RIS_LOS=I_phi*atand(abs(y_RIS-y_Tx)/abs(x_RIS-x_Tx));
        I_theta=sign(z_Tx-z_RIS);
        theta_T_RIS_LOS=I_theta*asind(abs(z_RIS-z_Tx)/d_T_RIS);
        I_phi_Tx=sign(y_Tx-y_RIS);
        phi_Tx_LOS=I_phi_Tx*atand(abs(y_RIS-y_Tx)/abs(x_RIS-x_Tx));
        I_theta_Tx=sign(z_RIS-z_Tx);
        theta_Tx_LOS=I_theta_Tx*asind(abs(z_RIS-z_Tx)/d_T_RIS);
    end
    array_RIS_LOS=zeros(1,N);
    counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_RIS_LOS(counter3)=exp(1i*k*dis*(x*sind(theta_T_RIS_LOS)+y*sind(phi_T_RIS_LOS)*cosd(theta_T_RIS_LOS)));
            counter3=counter3+1;
        end
    end
    if ArrayType==1
        array_Tx_LOS=zeros(1,Nt); counter3=1;
        for x=0:Nt-1
            array_Tx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_LOS)*cosd(theta_Tx_LOS)));
            counter3=counter3+1;
        end
    else
        counter3=1;
        for x=0:sqrt(Nt)-1
            for y=0:sqrt(Nt)-1
                array_Tx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_LOS)*cosd(theta_Tx_LOS)+y*sind(theta_Tx_LOS)));
                counter3=counter3+1;
            end
        end
    end
    L_dB_LOS=-20*log10(4*pi/lambda)-10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_T_RIS)-randn*sigma_LOS;
    L_LOS=10^(L_dB_LOS/10);
    h_LOS=sqrt(L_LOS)*transpose(array_RIS_LOS)*array_Tx_LOS*exp(1i*rand*2*pi)*sqrt(Gain*(cosd(theta_T_RIS_LOS))^(2*q));
else
    h_LOS=0;
end

for generate=1:100
    C=max([1,poissrnd(lambda_p)]);
    S=randi(30,1,C);
    phi_Tx=[]; theta_Tx=[]; phi_av=zeros(1,C); theta_av=zeros(1,C);
    for counter=1:C
        phi_av(counter)=rand*180-90; theta_av(counter)=rand*90-45;
        phi_Tx=[phi_Tx,log(rand(1,S(counter))./rand(1,S(counter)))*sqrt(25/2)+phi_av(counter)];
        theta_Tx=[theta_Tx,log(rand(1,S(counter))./rand(1,S(counter)))*sqrt(25/2)+theta_av(counter)];
    end
    a_c=1+rand(1,C)*(d_T_RIS-1);
    if Environment==1
        dim=[75,50,3.5];
        Coordinates=zeros(C,3); Coordinates2=zeros(sum(S),3);
        for counter=1:C
            Coordinates(counter,:)=[x_Tx+a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                y_Tx-a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                z_Tx+a_c(counter)*sind(theta_av(counter))];
            while Coordinates(counter,3)>dim(3)||Coordinates(counter,3)<0||Coordinates(counter,2)>dim(2)||Coordinates(counter,2)<0||Coordinates(counter,1)>dim(1)||Coordinates(counter,1)<0
                a_c(counter)=0.8*a_c(counter);
                Coordinates(counter,:)=[x_Tx+a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                    y_Tx-a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                    z_Tx+a_c(counter)*sind(theta_av(counter))];
            end
        end
    else
        Coordinates=zeros(C,3); Coordinates2=zeros(sum(S),3);
        for counter=1:C
            Coordinates(counter,:)=[x_Tx+a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                y_Tx-a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                z_Tx+a_c(counter)*sind(theta_av(counter))];
            while Coordinates(counter,3)<0
                a_c(counter)=0.8*a_c(counter);
                Coordinates(counter,:)=[x_Tx+a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                    y_Tx-a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                    z_Tx+a_c(counter)*sind(theta_av(counter))];
            end
        end
    end
    a_c_rep=[];
    for counter3=1:C, a_c_rep=[a_c_rep,repmat(a_c(counter3),1,S(counter3))]; end
    for counter2=1:sum(S)
        Coordinates2(counter2,:)=[x_Tx+a_c_rep(counter2)*cosd(theta_Tx(counter2))*cosd(phi_Tx(counter2)),...
            y_Tx-a_c_rep(counter2)*cosd(theta_Tx(counter2))*sind(phi_Tx(counter2)),...
            z_Tx+a_c_rep(counter2)*sind(theta_Tx(counter2))];
    end
    if Environment==1
        ignore=[];
        for counter2=1:sum(S)
            if Coordinates2(counter2,3)>dim(3)||Coordinates2(counter2,3)<0||Coordinates2(counter2,2)>dim(2)||Coordinates2(counter2,2)<0||Coordinates2(counter2,1)>dim(1)||Coordinates2(counter2,1)<0
                ignore=[ignore,counter2]; end
        end
    else
        ignore=[];
        for counter2=1:sum(S)
            if Coordinates2(counter2,3)<0, ignore=[ignore,counter2]; end
        end
    end
    indices=setdiff(1:sum(S),ignore); M_new=length(indices);
    if M_new>0, break; end
end

phi_cs_RIS=zeros(1,sum(S)); theta_cs_RIS=zeros(1,sum(S));
phi_Tx_cs=zeros(1,sum(S));  theta_Tx_cs=zeros(1,sum(S));
b_cs=zeros(1,sum(S));       d_cs=zeros(1,sum(S));

if Scenario==1
    for counter2=indices
        b_cs(counter2)=norm(RIS_xyz-Coordinates2(counter2,:));
        d_cs(counter2)=a_c_rep(counter2)+b_cs(counter2);
        I_phi=sign(x_RIS-Coordinates2(counter2,1));
        phi_cs_RIS(counter2)=I_phi*atand(abs(x_RIS-Coordinates2(counter2,1))/abs(y_RIS-Coordinates2(counter2,2)));
        I_theta=sign(Coordinates2(counter2,3)-z_RIS);
        theta_cs_RIS(counter2)=I_theta*asind(abs(z_RIS-Coordinates2(counter2,3))/b_cs(counter2));
        I_phi_Tx_cs=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs(counter2)=I_phi_Tx_cs*atand(abs(Coordinates2(counter2,2)-y_Tx)/abs(Coordinates2(counter2,1)-x_Tx));
        I_theta_Tx_cs=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs(counter2)=I_theta_Tx_cs*asind(abs(Coordinates2(counter2,3)-z_Tx)/a_c_rep(counter2));
    end
else
    for counter2=indices
        b_cs(counter2)=norm(RIS_xyz-Coordinates2(counter2,:));
        d_cs(counter2)=a_c_rep(counter2)+b_cs(counter2);
        I_phi=sign(Coordinates2(counter2,2)-y_RIS);
        phi_cs_RIS(counter2)=I_phi*atand(abs(y_RIS-Coordinates2(counter2,2))/abs(x_RIS-Coordinates2(counter2,1)));
        I_theta=sign(Coordinates2(counter2,3)-z_RIS);
        theta_cs_RIS(counter2)=I_theta*asind(abs(z_RIS-Coordinates2(counter2,3))/b_cs(counter2));
        I_phi_Tx_cs=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs(counter2)=I_phi_Tx_cs*atand(abs(Coordinates2(counter2,2)-y_Tx)/abs(Coordinates2(counter2,1)-x_Tx));
        I_theta_Tx_cs=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs(counter2)=I_theta_Tx_cs*asind(abs(Coordinates2(counter2,3)-z_Tx)/a_c_rep(counter2));
    end
end

array_cs_RIS=zeros(sum(S),N);
for counter2=indices
    counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_cs_RIS(counter2,counter3)=exp(1i*k*dis*(x*sind(theta_cs_RIS(counter2))+y*sind(phi_cs_RIS(counter2))*cosd(theta_cs_RIS(counter2))));
            counter3=counter3+1;
        end
    end
end
array_Tx_cs=zeros(sum(S),Nt);
if ArrayType==1
    for counter2=indices
        counter3=1;
        for x=0:Nt-1
            array_Tx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs(counter2))*cosd(theta_Tx_cs(counter2))));
            counter3=counter3+1;
        end
    end
else
    for counter2=indices
        counter3=1;
        for x=0:sqrt(Nt)-1
            for y=0:sqrt(Nt)-1
                array_Tx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs(counter2))*cosd(theta_Tx_cs(counter2))+y*sind(theta_Tx_cs(counter2))));
                counter3=counter3+1;
            end
        end
    end
end

h_NLOS=zeros(N,Nt); beta=zeros(1,sum(S)); shadow=beta;
for counter2=indices
    X_sigma=randn*sigma_NLOS;
    Lcs_dB=-20*log10(4*pi/lambda)-10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs(counter2))-X_sigma;
    Lcs=10^(Lcs_dB/10);
    beta(counter2)=((randn+1i*randn)/sqrt(2));
    shadow(counter2)=X_sigma;
    h_NLOS=h_NLOS+beta(counter2)*sqrt(Gain*(cosd(theta_cs_RIS(counter2)))^(2*q))*sqrt(Lcs)*transpose(array_cs_RIS(counter2,:))*array_Tx_cs(counter2,:);
end
h_NLOS=h_NLOS*sqrt(1/M_new);
h=h_NLOS+h_LOS;

if Environment==1
    d_RIS_R=norm(RIS_xyz-Rx_xyz);
    I_theta=sign(z_Rx-z_RIS);
    theta_Rx_RIS=I_theta*asind(abs(z_Rx-z_RIS)/d_RIS_R);
    if Scenario==1
        I_phi=sign(x_RIS-x_Rx);
        phi_Rx_RIS=I_phi*atand(abs(x_Rx-x_RIS)/abs(y_Rx-y_RIS));
    else
        I_phi=sign(y_Rx-y_RIS);
        phi_Rx_RIS=I_phi*atand(abs(y_Rx-y_RIS)/abs(x_Rx-x_RIS));
    end
    phi_av_Rx=rand*180-90; theta_av_Rx=rand*180-90;
    phi_Rx=log(rand/rand)*sqrt(25/2)+phi_av_Rx;
    theta_Rx=log(rand/rand)*sqrt(25/2)+theta_av_Rx;
    array_2=zeros(1,N); counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_2(counter3)=exp(1i*k*dis*(x*sind(theta_Rx_RIS)+y*sind(phi_Rx_RIS)*cosd(theta_Rx_RIS)));
            counter3=counter3+1;
        end
    end
    array_Rx=zeros(1,Nr);
    if ArrayType==1
        counter3=1;
        for x=0:Nr-1
            array_Rx(counter3)=exp(1i*k*dis*(x*sind(phi_Rx)*cosd(theta_Rx)));
            counter3=counter3+1;
        end
    else
        counter3=1;
        for x=0:sqrt(Nr)-1
            for y=0:sqrt(Nr)-1
                array_Rx(counter3)=exp(1i*k*dis*(x*sind(phi_Rx)*cosd(theta_Rx)+y*sind(theta_Rx)));
                counter3=counter3+1;
            end
        end
    end
    L_dB_LOS_2=-20*log10(4*pi/lambda)-10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_RIS_R)-randn*sigma_LOS;
    L_LOS_2=10^(L_dB_LOS_2/10);
    g=sqrt(Gain*(cosd(theta_Rx_RIS))^(2*q))*sqrt(L_LOS_2)*transpose(array_2)*array_Rx*exp(1i*rand*2*pi);

else
    d_RIS_R=norm(RIS_xyz-Rx_xyz);
    p_LOS_2=min([20/d_RIS_R,1])*(1-exp(-d_RIS_R/39))+exp(-d_RIS_R/39);
    I_LOS_2=randsrc(1,1,[1,0;p_LOS_2 1-p_LOS_2]);
    if I_LOS_2==1
        if Scenario==1
            I_phi=sign(x_RIS-x_Rx); phi_RIS_R_LOS=I_phi*atand(abs(x_RIS-x_Rx)/abs(y_RIS-y_Rx));
            I_theta=sign(z_Rx-z_RIS); theta_RIS_R_LOS=I_theta*asind(abs(z_RIS-z_Rx)/d_RIS_R);
        else
            I_phi=sign(y_Rx-y_RIS); phi_RIS_R_LOS=I_phi*atand(abs(y_RIS-y_Rx)/abs(x_RIS-x_Rx));
            I_theta=sign(z_Rx-z_RIS); theta_RIS_R_LOS=I_theta*asind(abs(z_RIS-z_Rx)/d_RIS_R);
        end
        phi_av_Rx_LOS=rand*180-90; theta_av_Rx_LOS=rand*180-90;
        phi_Rx_LOS=log(rand/rand)*sqrt(25/2)+phi_av_Rx_LOS;
        theta_Rx_LOS=log(rand/rand)*sqrt(25/2)+theta_av_Rx_LOS;
        array_RIS_Rx_LOS=zeros(1,N); array_Rx_LOS=zeros(1,Nr);
        counter3=1;
        for x=0:sqrt(N)-1
            for y=0:sqrt(N)-1
                array_RIS_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(theta_RIS_R_LOS)+y*sind(phi_RIS_R_LOS)*cosd(theta_RIS_R_LOS)));
                counter3=counter3+1;
            end
        end
        if ArrayType==1
            counter3=1;
            for x=0:Nr-1
                array_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_LOS)*cosd(theta_Rx_LOS)));
                counter3=counter3+1;
            end
        else
            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_LOS)*cosd(theta_Rx_LOS)+y*sind(theta_Rx_LOS)));
                    counter3=counter3+1;
                end
            end
        end
        L_dB_LOS_2=-20*log10(4*pi/lambda)-10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_RIS_R)-randn*sigma_LOS;
        L_LOS_2=10^(L_dB_LOS_2/10);
        g_LOS=sqrt(L_LOS_2)*transpose(array_RIS_Rx_LOS)*array_Rx_LOS*exp(1i*rand*2*pi)*sqrt(Gain*(cosd(theta_RIS_R_LOS))^(2*q));
    else
        g_LOS=0;
    end
    for generate2=1:100
        C_2=max([1,poissrnd(lambda_p)]); S_2=randi(30,1,C_2);
        phi_Tx_2=[]; theta_Tx_2=[]; phi_av_2=zeros(1,C_2); theta_av_2=zeros(1,C_2);
        for counter=1:C_2
            phi_av_2(counter)=rand*90-45; theta_av_2(counter)=rand*90-45;
            phi_Tx_2=[phi_Tx_2,log(rand(1,S_2(counter))./rand(1,S_2(counter)))*sqrt(25/2)+phi_av_2(counter)];
            theta_Tx_2=[theta_Tx_2,log(rand(1,S_2(counter))./rand(1,S_2(counter)))*sqrt(25/2)+theta_av_2(counter)];
        end
        a_c_2=1+rand(1,C_2)*(d_RIS_R-1);
        Coordinates_2=zeros(C_2,3); Coordinates2_2=zeros(sum(S_2),3);
        for counter=1:C_2
            if Scenario==1
                Coordinates_2(counter,:)=[x_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),y_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),z_RIS+a_c_2(counter)*sind(theta_av_2(counter))];
            else
                Coordinates_2(counter,:)=[x_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),y_RIS+a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),z_RIS+a_c_2(counter)*sind(theta_av_2(counter))];
            end
            while Coordinates_2(counter,3)<0
                a_c_2(counter)=0.8*a_c_2(counter);
                if Scenario==1
                    Coordinates_2(counter,:)=[x_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),y_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),z_RIS+a_c_2(counter)*sind(theta_av_2(counter))];
                else
                    Coordinates_2(counter,:)=[x_RIS-a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),y_RIS+a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),z_RIS+a_c_2(counter)*sind(theta_av_2(counter))];
                end
            end
        end
        a_c_rep_2=[];
        for counter3=1:C_2, a_c_rep_2=[a_c_rep_2,repmat(a_c_2(counter3),1,S_2(counter3))]; end
        for counter2=1:sum(S_2)
            if Scenario==1
                Coordinates2_2(counter2,:)=[x_RIS-a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*sind(phi_Tx_2(counter2)),y_RIS-a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*cosd(phi_Tx_2(counter2)),z_RIS+a_c_rep_2(counter2)*sind(theta_Tx_2(counter2))];
            else
                Coordinates2_2(counter2,:)=[x_RIS-a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*cosd(phi_Tx_2(counter2)),y_RIS+a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*sind(phi_Tx_2(counter2)),z_RIS+a_c_rep_2(counter2)*sind(theta_Tx_2(counter2))];
            end
        end
        ignore_2=[];
        for counter2=1:sum(S_2)
            if Coordinates2_2(counter2,3)<0, ignore_2=[ignore_2,counter2]; end
        end
        indices_2=setdiff(1:sum(S_2),ignore_2); M_new_2=length(indices_2);
        if M_new_2>0, break; end
    end
    array_2=zeros(sum(S_2),N);
    for counter2=indices_2
        counter3=1;
        for x=0:sqrt(N)-1
            for y=0:sqrt(N)-1
                array_2(counter2,counter3)=exp(1i*k*dis*(x*sind(theta_Tx_2(counter2))+y*sind(phi_Tx_2(counter2))*cosd(theta_Tx_2(counter2))));
                counter3=counter3+1;
            end
        end
    end
    b_cs_2=zeros(1,sum(S_2)); d_cs_2=zeros(1,sum(S_2));
    phi_cs_Rx=zeros(1,sum(S_2)); theta_cs_Rx=zeros(1,sum(S_2));
    for counter2=indices_2
        b_cs_2(counter2)=norm(Rx_xyz-Coordinates2_2(counter2,:));
        d_cs_2(counter2)=a_c_rep_2(counter2)+b_cs_2(counter2);
        phi_av_Rx_NLOS=rand*180-90; theta_av_Rx_NLOS=rand*180-90;
        phi_cs_Rx(counter2)=log(rand/rand)*sqrt(25/2)+phi_av_Rx_NLOS;
        theta_cs_Rx(counter2)=log(rand/rand)*sqrt(25/2)+theta_av_Rx_NLOS;
    end
    array_Rx_cs=zeros(sum(S_2),Nr);
    if ArrayType==1
        for counter2=indices_2
            counter3=1;
            for x=0:Nr-1
                array_Rx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx(counter2))*cosd(theta_cs_Rx(counter2))));
                counter3=counter3+1;
            end
        end
    else
        for counter2=indices_2
            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx(counter2))*cosd(theta_cs_Rx(counter2))+y*sind(theta_cs_Rx(counter2))));
                    counter3=counter3+1;
                end
            end
        end
    end
    g_NLOS=zeros(N,Nr);
    for counter2=indices_2
        X_sigma_2=randn*sigma_NLOS;
        Lcs_dB_2=-20*log10(4*pi/lambda)-10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs_2(counter2))-X_sigma_2;
        Lcs_2=10^(Lcs_dB_2/10);
        beta_2=((randn+1i*randn)/sqrt(2));
        g_NLOS=g_NLOS+beta_2*sqrt(Lcs_2)*sqrt(Gain*(cosd(theta_Tx_2(counter2)))^(2*q))*transpose(array_2(counter2,:))*array_Rx_cs(counter2,:);
    end
    g_NLOS=g_NLOS*sqrt(1/M_new_2);
    g=g_NLOS+g_LOS;
end

% h_SISO generation (Environment 1)
if Environment==1
    d_T_R=norm(Tx_xyz-Rx_xyz);
    d_cs_tilde=zeros(1,sum(S)); h_SISO_NLOS=0;
    for counter2=indices
        d_cs_tilde(counter2)=a_c_rep(counter2)+norm(Coordinates2(counter2,:)-Rx_xyz);
        I_phi_Tx_cs_SISO=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs_SISO(counter2)=I_phi_Tx_cs_SISO*atand(abs(Coordinates2(counter2,2)-y_Tx)/abs(Coordinates2(counter2,1)-x_Tx));
        I_theta_Tx_cs_SISO=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs_SISO(counter2)=I_theta_Tx_cs_SISO*asind(abs(Coordinates2(counter2,3)-z_Tx)/a_c_rep(counter2));
        phi_av_SISO=rand*180-90; theta_av_SISO=rand*180-90;
        phi_cs_Rx_SISO(counter2)=log(rand/rand)*sqrt(25/2)+phi_av_SISO;
        theta_cs_Rx_SISO(counter2)=log(rand/rand)*sqrt(25/2)+theta_av_SISO;
        if ArrayType==1
            counter3=1;
            for x=0:Nr-1
                array_Rx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx_SISO(counter2))*cosd(theta_cs_Rx_SISO(counter2))));
                counter3=counter3+1;
            end
            counter3=1;
            for x=0:Nt-1
                array_Tx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs_SISO(counter2))*cosd(theta_Tx_cs_SISO(counter2))));
                counter3=counter3+1;
            end
        else
            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx_SISO(counter2))*cosd(theta_cs_Rx_SISO(counter2))+y*sind(theta_cs_Rx_SISO(counter2))));
                    counter3=counter3+1;
                end
            end
            counter3=1;
            for x=0:sqrt(Nt)-1
                for y=0:sqrt(Nt)-1
                    array_Tx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs_SISO(counter2))*cosd(theta_Tx_cs_SISO(counter2))+y*sind(theta_Tx_cs_SISO(counter2))));
                    counter3=counter3+1;
                end
            end
        end
        Lcs_dB_SISO=-20*log10(4*pi/lambda)-10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs_tilde(counter2))-shadow(counter2);
        Lcs_SISO=10^(Lcs_dB_SISO/10);
        eta=k*(norm(Coordinates2(counter2,:)-RIS_xyz)-norm(Coordinates2(counter2,:)-Rx_xyz));
        h_SISO_NLOS=h_SISO_NLOS+beta(counter2)*exp(1i*eta)*sqrt(Lcs_SISO)*transpose(array_Rx_cs_SISO(counter2,:))*array_Tx_cs_SISO(counter2,:);
    end
    h_SISO_NLOS=h_SISO_NLOS*sqrt(1/M_new);
    if z_RIS>=z_Tx
        if d_T_R<=1.2,       p_LOS_3=1;
        elseif d_T_R<6.5,    p_LOS_3=exp(-(d_T_R-1.2)/4.7);
        else,                 p_LOS_3=0.32*exp(-(d_T_R-6.5)/32.6); end
        I_LOS_3=randsrc(1,1,[1,0;p_LOS_3 1-p_LOS_3]);
    else
        I_LOS_3=I_LOS;
    end
    if I_LOS_3==1
        L_SISO_LOS_dB=-20*log10(4*pi/lambda)-10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_T_R)-randn*sigma_LOS;
        L_SISO_LOS=10^(L_SISO_LOS_dB/10);
        I_phi_Tx_SISO=sign(y_Tx-y_Rx); phi_Tx_SISO=I_phi_Tx_SISO*atand(abs(y_Tx-y_Rx)/abs(x_Tx-x_Rx));
        I_theta_Tx_SISO=sign(z_Rx-z_Tx); theta_Tx_SISO=I_theta_Tx_SISO*atand(abs(z_Rx-z_Tx)/abs(d_T_R));
        phi_av_S=rand*180-90; theta_av_S=rand*180-90;
        phi_Rx_SISO=log(rand/rand)*sqrt(25/2)+phi_av_S;
        theta_Rx_SISO=log(rand/rand)*sqrt(25/2)+theta_av_S;
        if ArrayType==1
            counter3=1;
            for x=0:Nt-1, array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO))); counter3=counter3+1; end
            counter3=1;
            for x=0:Nr-1, array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO))); counter3=counter3+1; end
        else
            counter3=1;
            for x=0:sqrt(Nt)-1, for y=0:sqrt(Nt)-1, array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO)+y*sind(theta_Tx_SISO))); counter3=counter3+1; end, end
            counter3=1;
            for x=0:sqrt(Nr)-1, for y=0:sqrt(Nr)-1, array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO)+y*sind(theta_Rx_SISO))); counter3=counter3+1; end, end
        end
        h_SISO_LOS=sqrt(L_SISO_LOS)*exp(1i*rand*2*pi)*transpose(array_Rx_SISO)*array_Tx_SISO;
    else
        h_SISO_LOS=0;
    end
    h_SISO=h_SISO_NLOS+h_SISO_LOS;
else
    h_SISO=zeros(Nr,Nt);  % placeholder for outdoor (not used by JDCNet)
end

end
