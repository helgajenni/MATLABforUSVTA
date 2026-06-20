%% ============================================================
% USV 4-DOF RRT* + G2CBS + ILOS + PID + LMPC
% PHASE 2: DYNAMIC OBSTACLE AVOIDANCE WITH LINEAR MPC
% ─────────────────────────────────────────────────────────────
% Arsitektur (Opsi B — Path Replanning via LMPC):
%
%   NORMAL MODE  : fullPath → ILOS → psi_d → PID → aktuator
%   AVOIDANCE    : LMPC prediksi obs dyn N langkah ke depan
%                  → optimize [psi_d, U0] meminimalkan deviasi
%                    dari fullPath SAMBIL jaga jarak obs dyn
%                  → output step pertama → PID (Receding Horizon)
%   RETURN MODE  : jarak obs dyn > r_clear → kembali ke ILOS
%
% Referensi:
%   [1] Yuan et al. (2022), Math. Prob. Eng. — MPC + CVM prediction
%   [2] Gonzalez-Garcia et al. (2022), Ocean Eng. — NMPC path-follow
%   [3] Lee et al. (2022), IFAC — Tube-MPC surface vehicle
% ─────────────────────────────────────────────────────────────
% Model LMPC: linearisasi kinematik planar di sekitar U0=1.5 m/s
%   x_{k+1} = A*x_k + B*u_k
%   state  : [xe, ye, psi_e]  (error dari referensi path di wpIdx)
%   input  : [delta_psi, delta_U]  (koreksi heading & speed)
%
% Constraint safety: ||pos_USV_k - obs_pred_k|| >= r_safe
%   Dilinearisasi sebagai half-plane constraint (lihat lmpc_solve)
%% ============================================================
clear; clc; close all;

%% ===================== SEED SETUP ===============================
seed_value = 50;
rng(seed_value);
fprintf('=== Phase 2: LMPC Dynamic Obstacle Avoidance ===\n');
fprintf('=== Seed: %d ===\n', seed_value);

%% ===================== MAP & OBSTACLE SETUP =====================
mapSize  = [33 50];
xMax     = mapSize(2);
yMax     = mapSize(1);

obstacles = [10  10   0.25;
             20  20   0.25;
             30  10   0.25;
             40  20   0.25;
             17  16.5 0.25;
             41  16   0.25];

start    = [1,  8];
waypoint = [25, 20.0];
goal     = [48, 13];

safetyMargin  = 1.5;
displayMargin = 1.0;

%% =================== DYNAMIC OBSTACLES ==========================
% Format: [x0, y0, radius, vx, vy]
obs_dyn = [13.0  7.6  0.25   0.0  +0.5;   % D1: bawah→atas, Seg1 ~t=14s
           30.0  28.0  0.25   0.0  -0.28];  % D2: atas→bawah, Seg2 ~t=34s

n_dyn    = size(obs_dyn, 1);
r_detect = 5.0;  % [m] radius mulai aktifkan LMPC — cukup waktu reaksi
r_safe   =  1.0;  % [m] minimum jarak aman (obs_r + USV_r + margin)
r_collision = 0.3;
r_clear  = 2.8;  % [m] radius kembali ke ILOS

%% =================== LMPC PARAMETERS ===========================
lmpc.N        = 10;      % Prediction horizon (N*dt=0.5s) — ringan, cukup untuk react
lmpc.dt       = 0.05;    % sama dengan dt simulasi
lmpc.U0_nom   = 1.5;     % kecepatan nominal [m/s]

% Bobot cost function
% Q: penalti deviasi dari path referensi [xe^2, ye^2, psi_e^2]
% R: penalti control effort [delta_psi^2, delta_U^2]
% Qf: terminal cost (lebih besar dari Q untuk ketat kembali ke path)
lmpc.Q  = diag([1.0,  30.0,  1.0]);   % ye dipenalti kuat
lmpc.R  = diag([1.0,   2.0]);          % penalti input sedang
lmpc.Qf = diag([2.0,  60.0,  2.0]);   % terminal cost
% Constraint input
lmpc.dpsi_max = deg2rad(12);   % max heading correction — cukup untuk belok ke depan
lmpc.dU_max   = 0.8;           % max speed correction [m/s]
lmpc.U_min    = 0.4;           % minimum speed saat avoidance [m/s]
lmpc.U_max    = 1.5;           % maximum speed

fprintf('  LMPC: N=%d, horizon=%.1fs, r_detect=%.1fm, r_safe=%.1fm\n', ...
    lmpc.N, lmpc.N*lmpc.dt, r_detect, r_safe);
blend.tau_engage  = 0.3;   % tetap
blend.tau_release = 1.0;   % turun dari 2.5 → release lebih cepat
blend.d_enter     = 4.0;   % turun dari 4.8 → hanya engage saat benar-benar dekat
blend.d_exit      = 5.5;   % turun dari 7.5 → jangan nempel lama
blend.d_safe_lock = 1.2;   % turun dari 2.5 → hanya kritis saat sangat dekat
blend.w_min_lmpc  = 0.05;
blend.w_max_lmpc  = 0.95;
blend.w           = 0;
%% =================== USV PARAMETERS ============================
USV.L=1.6; USV.B=0.4; USV.m=11.8; USV.g=9.81;
USV.A1=1.5066; USV.A2=-0.7405; USV.A3=0.4219; USV.A4=-0.1397;
USV.A5=-0.1464; USV.A6=-3.1952; USV.A7=4.1189;
USV.A8=0; USV.A9=0; USV.A10=0.0845; USV.A11=0.0561;
USV.A12=-1.0495; USV.A13=1.4038; USV.A14=-2.0764;
USV.A15=0.001; USV.A16=0.9671; USV.A17=0.0021;
USV.A18=0.0178; USV.A19=0.001;
USV.A20=0; USV.A21=0; USV.A22=0;
USV.K.KpLin=0; USV.K.KpAbs=0; USV.K.KpCub=0;
USV.K.Kphi=13.5523; USV.K.Kfy=-0.0175; USV.K.Kv=-3.3096;
USV.K.Kr=-2.7576; USV.K.Kdelta=0.1738; USV.K.Kbias=-0.3631;

lims.TX=200; lims.TY=60; lims.TN=1750; lims.TK=0.7;
lims.dTX=200; lims.dTN=7500; lims.dTK=5;

%% =================== RRT* ======================================
rrt.maxIter=800; rrt.stepSize=1.5; rrt.goalBias=0.15;
rrt.rewireRad=3.0; rrt.goalTol=1.0;

fprintf('\n=== Menjalankan RRT* ===\n');
tic; rawPath1=rrtStar(start,waypoint,obstacles,mapSize,rrt,safetyMargin); waktu_rrt1=toc;
if isempty(rawPath1), error('RRT* Seg1 failed'); end
tic; rawPath2=rrtStar(waypoint,goal,obstacles,mapSize,rrt,safetyMargin); waktu_rrt2=toc;
if isempty(rawPath2), error('RRT* Seg2 failed'); end
rawPathAll = [rawPath1; rawPath2(2:end,:)];

%% =================== PATH PROCESSING ===========================
rawRepaired    = repairPathObstacles(rawPathAll, obstacles, safetyMargin);
rawShortcut    = shortcutPath(rawRepaired, obstacles, safetyMargin, 800, 8.0);
rawDownsampled = downsamplePath(rawShortcut, 2.5);
rawChaikin     = chaikinSmooth(rawDownsampled, 4);
[~,idxClosest] = min(sqrt(sum((rawChaikin-waypoint).^2,2)));
shiftVec = waypoint - rawChaikin(idxClosest,:);
sigma = 8.0;
for i=1:size(rawChaikin,1)
    w = exp(-(abs(i-idxClosest)^2)/(2*sigma^2));
    rawChaikin(i,:) = rawChaikin(i,:) + shiftVec*w;
end
fullPath   = smooth_path_g2cbs_c2(rawChaikin, 60, 0);
[~,wpJunctionIdx] = min(sqrt(sum((fullPath-waypoint).^2,2)));
nWP        = size(fullPath,1);
pathCurvRaw= computePathCurvature(fullPath);
win        = max(1,min(500,floor(nWP/4)));
pathCurv   = conv(abs(pathCurvRaw)',ones(1,win)/win,'same')';
max_kappa  = max(abs(pathCurv));
seg1Len    = wpJunctionIdx;
smoothPath1= fullPath(1:wpJunctionIdx,:);
smoothPath2= fullPath(wpJunctionIdx:end,:);
fprintf('  Path OK. nWP=%d, max_kappa=%.4f\n', nWP, max_kappa);

%% =================== CONTROLLER PARAMETERS =====================
ILOS.Delta=2.4; ILOS.gamma=0.2; ILOS.beta_hat=0; ILOS.kappa=0.25;
WP_RADIUS=1.0; GOAL_TOL=2.0;
gains.Ku_p=60; gains.Ku_i=0.5; gains.Ku_d=10;
gains.Kpsi_p=5550; gains.Kpsi_i=95; gains.Kpsi_d=3550;
gains.Kphi_p=120; gains.Kphi_i=1; gains.Kphi_d=80;
filt.alpha_r=0.8; filt.alpha_p=0.8; filt.r_filt=0; filt.p_filt=0;
U0_target=1.5; U0_curv_speed=0.8; U0_saved=0; alpha_u=0.45;

%% =================== SIMULATION SETUP ==========================
dt=0.05; Tsim=200; N=round(Tsim/dt);

wpIdx=2;
idxNext_init=min(wpIdx+12,nWP);
psi0=atan2(fullPath(idxNext_init,2)-fullPath(wpIdx,2),...
           fullPath(idxNext_init,1)-fullPath(wpIdx,1));

state = zeros(N+1,8);
state(1,:) = [start(1),start(2),psi0,0,0,0,0,0];

%% Log standar
log.t=zeros(N+1,1); log.psi_d=zeros(N+1,1); log.psi_e=zeros(N+1,1);
log.cte=zeros(N+1,1); log.u=zeros(N+1,1); log.v=zeros(N+1,1);
log.r=zeros(N+1,1); log.p=zeros(N+1,1); log.phi=zeros(N+1,1);
log.TN=zeros(N+1,1); log.pathSeg=ones(N+1,1);

%% Log LMPC khusus
log.dyn_pos   = zeros(N+1,n_dyn*2);
log.dist_dyn  = zeros(N+1,n_dyn);
log.mode      = zeros(N+1,1);   % 0=ILOS, 1=LMPC avoidance
log.lmpc_psi  = zeros(N+1,1);   % psi_d dari LMPC
log.lmpc_U0   = zeros(N+1,1);   % U0 dari LMPC
log.collision = false(N+1,n_dyn);
collision_event = struct('time',{},'obs_id',{},'dist',{},'usv_pos',{},'obs_pos',{});

wpReached=false; N_end=N;
eInt_u=0; eInt_psi=0; eInt_phi=0;
intMax_u=25; intMax_psi=3; intMax_phi=1;
TX_prev=0; TN_prev=0; TK_prev=0;
psi_d_filtered=psi0; min_dist_to_goal=inf;
lmpc_active=false;
lmpc_active_id=0;
lmpc_on_time=0;       % waktu LMPC mulai aktif
lmpc_min_dur=3.0;    % minimum durasi — dikurangi agar lebih responsif
lmpc_max_dur=4.5;    % maksimum durasi — paksa OFF setelah 6s
lmpc_blocked = zeros(1, n_dyn);  % flag: obs_id yang sudah pernah dihindari
lmpc_blocked_dist = zeros(1, n_dyn);  % jarak saat LMPC OFF untuk obs itu
lmpc_passed = zeros(1, n_dyn); 
blend.w         = 0;
blend.w_log     = zeros(N+1, 1);
psi_d_ilos_filt = psi0;
psi_d_lmpc_last = psi0;
fprintf('\n=== Simulasi Mulai (DENGAN LMPC) ===\n');

%% =================== MAIN SIMULATION LOOP ======================
for k=1:N
    t  =(k-1)*dt;
    xk =state(k,1); yk =state(k,2);
    psi=state(k,3); phi=state(k,4);
    u  =state(k,5); v  =state(k,6);
    r  =state(k,7); p  =state(k,8);

    if any(isnan(state(k,:)))||any(isinf(state(k,:)))
        fprintf('  !!! NaN/Inf t=%.1fs\n',t); N_end=max(2,k-1); break;
    end

    %% ── UPDATE POSISI OBS DYN ────────────────────────────────────
    for id=1:n_dyn
        obs_dyn_state(id,1) = obs_dyn(id,1) + t*obs_dyn(id,4);
        obs_dyn_state(id,2) = obs_dyn(id,2) + t*obs_dyn(id,5);
        log.dyn_pos(k,(id-1)*2+1) = obs_dyn_state(id,1);
        log.dyn_pos(k,(id-1)*2+2) = obs_dyn_state(id,2);
    end
    % Reset blocked flag jika obs sudah sangat jauh — loop terpisah
    for id=1:n_dyn
        if lmpc_blocked(id)
            obs_x_now = obs_dyn(id,1) + t*obs_dyn(id,4);
            obs_y_now = obs_dyn(id,2) + t*obs_dyn(id,5);
            obs_ahead_id = cos(psi)*(obs_x_now-xk) + sin(psi)*(obs_y_now-yk);
            dist_now_id  = sqrt((xk-obs_x_now)^2 + (yk-obs_y_now)^2);
            % Tandai "passed" saat obs pertama kali ada di belakang
            if obs_ahead_id < 0
                lmpc_passed(id) = 1;
            end
            % Reset blocked hanya jika: sudah passed DAN jarak cukup jauh
            if lmpc_passed(id) && dist_now_id > r_detect * 1.5
                lmpc_blocked(id) = false;
                lmpc_passed(id)  = 0;
            end
        end
    end

    %% ── CEK JARAK (logging + collision check) ────────────────────
    for id=1:n_dyn
        d=norm([xk-obs_dyn_state(id,1), yk-obs_dyn_state(id,2)]);
        log.dist_dyn(k,id)=d;
        if d<(obs_dyn(id,3)+r_collision)
            log.collision(k,id)=true;
            if isempty(collision_event)||~any([collision_event.obs_id]==id)||...
               (t-collision_event(end).time)>1.0
                ev.time=t; ev.obs_id=id; ev.dist=d;
                ev.usv_pos=[xk,yk]; ev.obs_pos=obs_dyn_state(id,:);
                collision_event(end+1)=ev; %#ok<AGROW>
                fprintf('  *** COLLISION! t=%.1fs ObsDyn%d dist=%.3fm ***\n',t,id,d);
            end
        end
    end

    %% ── GOAL CHECK ───────────────────────────────────────────────
    dist_goal=norm([xk-goal(1),yk-goal(2)]);
    if dist_goal<min_dist_to_goal, min_dist_to_goal=dist_goal; end
    if dist_goal<GOAL_TOL
        fprintf('  >>> TARGET TERCAPAI t=%.1fs\n',t);
        state(k+1,5:8)=0; state(k,5:8)=0;
        state(k+1,1:2)=[goal(1),goal(2)];
        state(k+1,3)=psi; state(k+1,4)=0;
        log.t(k+1)=t; N_end=k+1; break;
    end

    %% ── WAYPOINT CHECK ───────────────────────────────────────────
    dist_wp=norm([xk-waypoint(1),yk-waypoint(2)]);
    if ~wpReached && dist_wp<0.3
        wpReached=true; ILOS.beta_hat=0; eInt_psi=0; eInt_u=0;
    end

    %% ── ADVANCE WP INDEX ─────────────────────────────────────────
    % Normal advance
    searchRange=wpIdx:min(wpIdx+17,nWP);
    dists=sqrt((fullPath(searchRange,1)-xk).^2+(fullPath(searchRange,2)-yk).^2);
    [~,localIdx]=min(dists); newIdx=searchRange(localIdx);
    if newIdx>=wpIdx, wpIdx=newIdx; end
    % Koreksi: jika USV jauh dari path, reset wpIdx ke posisi terdekat
    dist_to_path = min(sqrt((fullPath(:,1)-xk).^2+(fullPath(:,2)-yk).^2));
    if dist_to_path > 3.0 && blend.w < 0.2
        [~, idx_global] = min(sqrt((fullPath(:,1)-xk).^2+(fullPath(:,2)-yk).^2));
        wpIdx = max(wpIdx, idx_global);
    end
    log.pathSeg(k)=1+(wpIdx>seg1Len);

    idxNext=min(wpIdx+8,nWP);
    if idxNext==wpIdx && wpIdx>1
        dx_path=fullPath(wpIdx,1)-fullPath(wpIdx-1,1);
        dy_path=fullPath(wpIdx,2)-fullPath(wpIdx-1,2);
    else
        dx_path=fullPath(idxNext,1)-fullPath(wpIdx,1);
        dy_path=fullPath(idxNext,2)-fullPath(wpIdx,2);
    end
    alpha=atan2(dy_path,dx_path);
%% ── CARI OBS DYN TERDEKAT ───────────────────────────────
    min_d_dyn  = inf;
    nearest_id = 0;
    for id = 1:n_dyn
        d = log.dist_dyn(k, id);
        if d < min_d_dyn
            min_d_dyn  = d;
            nearest_id = id;
        end
    end
    futureCurvature = max(abs(pathCurv(wpIdx:min(wpIdx+25,nWP))));
    %% ── HITUNG REFERENSI PATH (selalu) ──────────────────────
    p1  = fullPath(wpIdx, :);
    ye  = -sin(alpha)*(xk - p1(1)) + cos(alpha)*(yk - p1(2));
    cte = ye;

   %% ── LANGKAH 1: HITUNG w_target ──────────────────────────
    % Guard utama: LMPC HANYA boleh aktif jika obs dinamis
    % benar-benar dalam radius deteksi r_detect.
    % Ini mencegah LMPC aktif saat obs masih jauh (6.84m tadi).
    obs_dyn_in_range = false;
    for id_chk = 1:n_dyn
        if log.dist_dyn(k, id_chk) < r_detect && ~lmpc_blocked(id_chk)
            obs_dyn_in_range = true;
            break;
        end
    end

    any_obs_critical = false;
    for id_chk = 1:n_dyn
        if log.dist_dyn(k, id_chk) < blend.d_safe_lock
            any_obs_critical = true; break;
        end
    end

    if ~obs_dyn_in_range
        % Tidak ada obs dalam jangkauan → paksa ILOS murni
        w_target = 0.0;
    elseif any_obs_critical
        % Obs dalam zona kritis → paksa LMPC penuh
        w_target = 1.0;
    elseif min_d_dyn < blend.d_enter && ~lmpc_blocked(nearest_id)
        w_target = 1.0;
    elseif min_d_dyn > blend.d_exit || lmpc_blocked(nearest_id)
        all_safe = true;
        for id_chk = 1:n_dyn
            if log.dist_dyn(k, id_chk) < blend.d_safe_lock
                all_safe = false; break;
            end
        end
        w_target = double(~all_safe);
    else
        ratio    = (min_d_dyn - blend.d_enter) / (blend.d_exit - blend.d_enter);
        ratio    = max(0, min(1, ratio));
        w_target = 1.0 - ratio*ratio*(3 - 2*ratio);
        if lmpc_blocked(nearest_id), w_target = 0.0; end
        if any_obs_critical, w_target = 1.0; end
    end

    %% ── LANGKAH 2: LOW-PASS ASIMETRIS ───────────────────────
    if w_target > blend.w
        tau_w = blend.tau_engage;
    else
        tau_w = blend.tau_release;
    end
    a_w     = dt / (tau_w + dt);
    blend.w = blend.w + a_w * (w_target - blend.w);
    blend.w = max(0.0, min(1.0, blend.w));
    blend.w_log(k) = blend.w;
    w = blend.w;

    %% ── LANGKAH 3: UPDATE LOG MODE ──────────────────────────
    lmpc_active    = (w > blend.w_min_lmpc);
    log.mode(k)    = lmpc_active;

    if lmpc_active && lmpc_active_id == 0
        lmpc_active_id = nearest_id;
        lmpc_on_time   = t;
        fprintf('  [BLEND ON ] t=%.1fs, ObsDyn%d, dist=%.2fm, w=%.2f\n', ...
            t, nearest_id, min_d_dyn, w);
    end

    % Mark blocked hanya jika obs benar-benar sudah aman (3 syarat)
    if w < blend.w_min_lmpc && lmpc_active_id ~= 0
        id_check      = lmpc_active_id;
        d_check       = log.dist_dyn(k, id_check);
        obs_x_chk     = obs_dyn_state(id_check,1);
        obs_y_chk     = obs_dyn_state(id_check,2);
        obs_ahead_chk = cos(psi)*(obs_x_chk-xk) + sin(psi)*(obs_y_chk-yk);
        cond_behind   = (obs_ahead_chk < 0);
        cond_safe     = (d_check > blend.d_safe_lock * 1.5);
        cond_far      = (d_check > blend.d_exit);
        if cond_behind && cond_safe && cond_far
            lmpc_blocked(id_check) = true;
            lmpc_passed(id_check)  = 0;
            lmpc_active_id         = 0;
            fprintf('  [BLEND OFF] t=%.1fs, blocked obs%d, dist=%.2fm\n', ...
                t, id_check, d_check);
        end
    end

    %% ── LANGKAH 4: ILOS (selalu dihitung di background) ─────
    if dist_goal < GOAL_TOL
        target_U0_ilos = 0;        current_Delta = 1.0;
    elseif dist_goal < WP_RADIUS
        target_U0_ilos = max(0, U0_target*sqrt(max(0,dist_goal-GOAL_TOL)/max(1e-3,WP_RADIUS)));
        current_Delta  = 1.0;
    elseif ~wpReached && dist_wp < 3.0
        target_U0_ilos = 0.8;      current_Delta = 0.6;
    elseif futureCurvature > 0.15
        target_U0_ilos = 0.8;      current_Delta = 0.8;
    elseif futureCurvature > 0.08
        target_U0_ilos = 0.9;      current_Delta = 1.2;
    else
        target_U0_ilos = U0_target; current_Delta = 1.8;
    end
    if ye > 2.0 && wpIdx > seg1Len
        target_U0_ilos = min(target_U0_ilos, 0.8);
        current_Delta  = min(current_Delta, 1.0);
    end

    if abs(ye) > 1.5 || w > 0.5
        % Freeze beta_hat saat LMPC dominan
        % — mencegah ILOS "ngacau" di background
        ILOS.beta_hat = 0;
    else
        ye_c = max(-current_Delta*1.2, min(current_Delta*1.2, ye));
        ILOS.beta_hat = ILOS.beta_hat + dt*ILOS.kappa*(ye_c/current_Delta);
        ILOS.beta_hat = max(-1, min(1, ILOS.beta_hat));
    end
    delta_eff       = max(current_Delta, abs(ye)*0.8);
    psi_ilos_raw    = wrapToPi(alpha - atan2(ye + ILOS.gamma*ILOS.beta_hat, delta_eff));
    ye_factor       = min(1.0, abs(ye)/2.0);
    alpha_lpf_ilos  = 0.20 + 0.26*ye_factor;
    psi_d_ilos_filt = psi_d_ilos_filt + alpha_lpf_ilos * wrapToPi(psi_ilos_raw - psi_d_ilos_filt);

    %% ── LANGKAH 5: LMPC (hanya jika w > threshold) ──────────
    target_U0_lmpc = U0_target;

    if w > blend.w_min_lmpc && lmpc_active_id > 0
        obs_pred = zeros(lmpc.N, 2);
        for i_pred = 1:lmpc.N
            obs_pred(i_pred,1) = obs_dyn_state(lmpc_active_id,1) + i_pred*dt*obs_dyn(lmpc_active_id,4);
            obs_pred(i_pred,2) = obs_dyn_state(lmpc_active_id,2) + i_pred*dt*obs_dyn(lmpc_active_id,5);
        end
        [psi_d_lmpc_raw, U0_lmpc] = lmpc_solve( ...
            [xk;yk;psi], u, fullPath, wpIdx, alpha, ...
            obs_pred, obs_dyn(lmpc_active_id,3), r_safe, lmpc, lims);
       alpha_lpf_lmpc  = 0.20;
        psi_d_lmpc_last = psi_d_lmpc_last + alpha_lpf_lmpc * ...
            wrapToPi(psi_d_lmpc_raw - psi_d_lmpc_last);

        % Batasi psi_lmpc dari psi_ilos secara adaptif:
        % Saat obs jauh (d > 3*r_safe) → batas ketat (15 deg) → CTE terjaga
        % Saat obs dekat (d < r_safe*2) → batas longgar (45 deg) → bisa menghindar
        d_to_active_obs = log.dist_dyn(k, lmpc_active_id);
        r_min_obs = obs_dyn(lmpc_active_id, 3) + r_safe;

        if d_to_active_obs < r_min_obs * 1.5
            max_psi_dev = deg2rad(50);   % zona sangat kritis — beri kebebasan penuh
        elseif d_to_active_obs < r_min_obs * 3.0
            max_psi_dev = deg2rad(30);   % zona bahaya — batas sedang
        else
            max_psi_dev = deg2rad(15);   % zona jauh — batas ketat, CTE terjaga
        end

        d_lmpc_vs_ilos = wrapToPi(psi_d_lmpc_last - psi_d_ilos_filt);
        if abs(d_lmpc_vs_ilos) > max_psi_dev
            psi_d_lmpc_last = wrapToPi(psi_d_ilos_filt + ...
                sign(d_lmpc_vs_ilos) * max_psi_dev);
        end

        target_U0_lmpc  = min(U0_lmpc, 1.2);
        log.lmpc_psi(k) = psi_d_lmpc_last;
        log.lmpc_U0(k)  = U0_lmpc;
    end

    %% ── LANGKAH 6: BLEND + SAFETY OVERRIDE ──────────────────
    % Safety override final: paksa w_eff=1 jika ada obs dalam zona kritis
    w_eff = w;
    if any_obs_critical, w_eff = 1.0; end

    d_heading      = atan2(sin(psi_d_lmpc_last - psi_d_ilos_filt), ...
                           cos(psi_d_lmpc_last - psi_d_ilos_filt));
    psi_d_filtered = wrapToPi(psi_d_ilos_filt + w_eff * d_heading);
    psi_d          = wrapToPi(psi_d_filtered);
    target_U0      = (1 - w_eff)*target_U0_ilos + w_eff*target_U0_lmpc;

    %% ── PID CONTROLLER ───────────────────────────────────────────
    psi_e  =wrapToPi(psi_d-psi);
    fade_in=min(1.0,t/4.0);

    if ~lmpc_active
        % Speed dari ILOS logic (sudah dihitung di atas)
        futureCurv2=max(abs(pathCurv(wpIdx:min(wpIdx+14,nWP))));
        if dist_goal<GOAL_TOL; target_U0=0;
        elseif dist_goal<WP_RADIUS
            target_U0=max(0,U0_target*sqrt(max(0,dist_goal-GOAL_TOL)/max(1e-3,WP_RADIUS)));
        elseif dist_goal<10
            target_U0=max(0,U0_target*sqrt(max(0,dist_goal-GOAL_TOL)/10));
        elseif futureCurv2>0.10; target_U0=U0_curv_speed;
        else; target_U0=U0_target;
        end
    end
    % Jika LMPC aktif, target_U0 sudah di-set dari lmpc_solve

    U0_saved=U0_saved+alpha_u*(target_U0-U0_saved);
    U0_saved=max(0, min(1.5, U0_saved));  % hard clamp speed reference    
    e_u  =U0_saved-u;
    e_phi=0-phi;

    eInt_u  =max(-intMax_u,  min(intMax_u,  eInt_u  +e_u*dt));
    eInt_psi=max(-intMax_psi,min(intMax_psi,eInt_psi+psi_e*dt));
    eInt_phi=max(-intMax_phi,min(intMax_phi,eInt_phi+e_phi*dt));
    if k>1 && psi_e*log.psi_e(k)<0, eInt_psi=eInt_psi*0.5; end

    filt.r_filt=filt.r_filt+filt.alpha_r*(r-filt.r_filt);
    filt.p_filt=filt.p_filt+filt.alpha_p*(p-filt.p_filt);

    u_ref=U0_saved;
    TX_ff=(0.7405*u_ref-0.4219*abs(u_ref)*u_ref+0.1397*(u_ref^2)*u_ref)/0.0178+gains.Ku_d*u_ref;
    TX=TX_ff+gains.Ku_p*e_u+gains.Ku_i*eInt_u-gains.Ku_d*u;

    % Saat jauh dari path (CTE besar), izinkan koreksi heading lebih besar
    if abs(ye) > 2.0 || abs(cte) > 2.0
        psi_e_c = max(-deg2rad(20), min(deg2rad(20), psi_e));
    else
        psi_e_c = max(-deg2rad(12), min(deg2rad(12), psi_e));
    end
    TN=gains.Kpsi_p*psi_e_c+gains.Kpsi_i*eInt_psi-gains.Kpsi_d*filt.r_filt;
    TK=gains.Kphi_p*e_phi+gains.Kphi_i*eInt_phi-gains.Kphi_d*filt.p_filt;
    TY=0;

    if dist_goal < GOAL_TOL * 1.5
        speed_factor = max(0, (dist_goal - GOAL_TOL) / (GOAL_TOL * 0.5));
        target_U0 = min(target_U0, U0_target * speed_factor);
    end
    if dist_goal < GOAL_TOL && u > 0.1
        TX = -150.0; TN = 0; TK = 0;
    elseif target_U0 <= 0.01 && dist_goal < 0.5
        TX = (u>0.15)*(-180); TN=0; TK=0;
    end

    TX=fade_in*max(-lims.TX,min(lims.TX,TX));
    TY=fade_in*max(-lims.TY,min(lims.TY,TY));
    TN=fade_in*max(-lims.TN,min(lims.TN,TN));
    TK=fade_in*max(-lims.TK,min(lims.TK,TK));

    TX=TX_prev+max(-lims.dTX*dt,min(lims.dTX*dt,TX-TX_prev));
    TN=TN_prev+max(-lims.dTN*dt,min(lims.dTN*dt,TN-TN_prev));
    TK=TK_prev+max(-lims.dTK*dt,min(lims.dTK*dt,TK-TK_prev));
    TX_prev=TX; TN_prev=TN; TK_prev=TK;

    %% ── DYNAMICS ─────────────────────────────────────────────────
    [Vdot,eta_dot]=usv4dof_python([u;v;r;p],[TX;TY;TN;TK],psi,phi,USV);
    state(k+1,1)=xk+eta_dot(1)*dt;
    state(k+1,2)=yk+eta_dot(2)*dt;
    state(k+1,3)=wrapToPi(psi+eta_dot(3)*dt);
    state(k+1,4)=max(-deg2rad(30),min(deg2rad(30),phi+eta_dot(4)*dt));
    state(k+1,5)=min(1.5,max(-5,min(5,u+Vdot(1)*dt)));
    state(k+1,6)=max(-0.6,min(0.6,v+Vdot(2)*dt));
    state(k+1,7)=max(-3,min(3,r+Vdot(3)*dt));
    state(k+1,8)=max(-5,min(5,p+Vdot(4)*dt));

    %% ── LOG ──────────────────────────────────────────────────────
    log.t(k+1)=t+dt; log.psi_d(k+1)=psi_d; log.psi_e(k+1)=psi_e;
    log.cte(k+1)=cte; log.u(k+1)=state(k+1,5); log.v(k+1)=state(k+1,6);
    log.r(k+1)=state(k+1,7); log.p(k+1)=state(k+1,8);
    log.phi(k+1)=state(k+1,4); log.TN(k+1)=TN;
    blend.w_log(k+1) = blend.w;
    N_end=k+1;
end

%% =================== SUMMARY ===================================
fprintf('\n=== HASIL SIMULASI ===\n');
fprintf('  Mode switches: %d kali LMPC aktif\n', sum(diff(log.mode(1:N_end))==1));
lmpc_steps=sum(log.mode(1:N_end));
fprintf('  Total LMPC aktif: %d steps (%.1fs)\n',lmpc_steps,lmpc_steps*dt);
fprintf('  Collision terjadi: %d events\n',length(collision_event));
if isempty(collision_event)
    fprintf('  ✓ SUKSES: Tidak ada collision!\n');
else
    fprintf('  ✗ Masih ada collision — perlu tuning parameter LMPC\n');
    for ev=collision_event
        fprintf('    t=%.1fs ObsDyn%d dist=%.3fm\n',ev.time,ev.obs_id,ev.dist);
    end
end
cte_act=log.cte(1:N_end);
fprintf('\n=== PATH TRACKING ERROR ===\n');
fprintf('  Max CTE : %.3f m\n',max(abs(cte_act)));
fprintf('  MAE CTE : %.3f m\n',mean(abs(cte_act)));
fprintf('  RMSE CTE: %.3f m\n',sqrt(mean(cte_act.^2)));

%% =================== FIGURE ====================================
Np=N_end; t_plot=log.t(1:Np); segs=log.pathSeg(1:Np);
theta_c=linspace(0,2*pi,60);
c1=[0.00 0.45 0.74]; c2=[0.85 0.33 0.10]; c3=[0.47 0.67 0.19]; c4=[0.49 0.18 0.56];
cObs=[0.8 0.1 0.1]; cDyn1=[1.0 0.5 0.0]; cDyn2=[0.5 0.0 0.8];
cTrail=[0.10 0.75 0.30];  % hijau terang — kontras dari biru & oranye path

hFig=figure('Name','Phase 2: LMPC Avoidance','Position',[20 20 1350 830]);
tg=uitabgroup(hFig);

%% TAB 1: Trajectory
tab1=uitab(tg,'Title',' Trajectory ');
ax1=axes('Parent',tab1,'Position',[0.05 0.08 0.90 0.86]);
hold(ax1,'on'); grid(ax1,'on'); axis(ax1,'equal');
xlim(ax1,[0 xMax]); ylim(ax1,[0 yMax]);
xlabel(ax1,'X [m]'); ylabel(ax1,'Y [m]');
title(ax1,'Phase 2: LMPC Dynamic Avoidance','FontWeight','bold','FontSize',12);

for i=1:size(obstacles,1)
    xc=obstacles(i,1); yc=obstacles(i,2); rc=obstacles(i,3)+displayMargin;
    fill(ax1,xc+rc*cos(theta_c),yc+rc*sin(theta_c),cObs,'FaceAlpha',.12,'EdgeColor',cObs);
    fill(ax1,xc+obstacles(i,3)*cos(theta_c),yc+obstacles(i,3)*sin(theta_c),cObs,'FaceAlpha',.5,'EdgeColor','none');
    text(ax1,xc,yc,sprintf('O%d',i),'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

% Global path smooth
plot(ax1,smoothPath1(:,1),smoothPath1(:,2),'-','Color',c1,'LineWidth',1.8,'DisplayName','Path Seg1');
plot(ax1,smoothPath2(:,1),smoothPath2(:,2),'-','Color',c2,'LineWidth',1.8,'DisplayName','Path Seg2');

% Trajectory USV — hijau terang
plot(ax1,state(1:Np,1),state(1:Np,2),'-','Color',cTrail,'LineWidth',2.8,'DisplayName','USV Trajectory');

% Highlight segmen LMPC aktif — warna merah muda
lmpc_idx=find(log.mode(1:Np)==1);
if ~isempty(lmpc_idx)
    plot(ax1,state(lmpc_idx,1),state(lmpc_idx,2),'o','Color',[1 0.2 0.2],...
        'MarkerSize',3,'DisplayName','LMPC active');
end

% Obs dyn trail
colors_dyn={cDyn1,cDyn2};
for id=1:n_dyn
    ox=log.dyn_pos(1:Np,(id-1)*2+1);
    oy=log.dyn_pos(1:Np,(id-1)*2+2);
    plot(ax1,ox,oy,':','Color',colors_dyn{id},'LineWidth',1.5,'DisplayName',sprintf('ObsDyn%d trail',id));
    plot(ax1,obs_dyn(id,1),obs_dyn(id,2),'v','MarkerSize',9,...
        'MarkerFaceColor',colors_dyn{id},'MarkerEdgeColor','k');
    t_end_sim=t_plot(end);
    oxe=obs_dyn(id,1)+t_end_sim*obs_dyn(id,4);
    oye=obs_dyn(id,2)+t_end_sim*obs_dyn(id,5);
    fill(ax1,oxe+obs_dyn(id,3)*cos(theta_c),oye+obs_dyn(id,3)*sin(theta_c),...
        colors_dyn{id},'FaceAlpha',0.4,'EdgeColor',colors_dyn{id});
end

plot(ax1,start(1),start(2),'^','MarkerSize',12,'MarkerFaceColor','g','MarkerEdgeColor','k');
plot(ax1,waypoint(1),waypoint(2),'s','MarkerSize',10,'MarkerFaceColor','b','MarkerEdgeColor','k');
plot(ax1,goal(1),goal(2),'p','MarkerSize',14,'MarkerFaceColor','r','MarkerEdgeColor','k');
legend(ax1,'Location','northwest','FontSize',8);

%% TAB 2: Mode Timeline
tab2=uitab(tg,'Title',' Mode Timeline ');
ax21=subplot(3,1,1,'Parent',tab2);
area(ax21,t_plot,log.mode(1:Np),'FaceColor',[1 0.8 0.8],'EdgeColor','none');
hold(ax21,'on');
plot(ax21,t_plot,log.mode(1:Np),'r-','LineWidth',1.5);
ylim(ax21,[-0.1 1.5]); xlabel(ax21,'t [s]'); ylabel(ax21,'Mode');
yticks(ax21,[0 1]); yticklabels(ax21,{'ILOS','LMPC'});
title(ax21,'Mode Switching (0=ILOS, 1=LMPC)','FontWeight','bold'); grid(ax21,'on');

ax22=subplot(3,1,2,'Parent',tab2);
for id=1:n_dyn
    plot(ax22,t_plot,log.dist_dyn(1:Np,id),'-','Color',colors_dyn{id},'LineWidth',1.8,...
        'DisplayName',sprintf('Dist D%d',id)); hold(ax22,'on');
end
yline(ax22,r_detect,'b--','LineWidth',1.5,'Label',sprintf('r_{detect}=%.1fm',r_detect));
yline(ax22,r_safe,'r--','LineWidth',1.5,'Label',sprintf('r_{safe}=%.1fm',r_safe));
yline(ax22,r_clear,'g--','LineWidth',1.2,'Label',sprintf('r_{clear}=%.1fm',r_clear));
xlabel(ax22,'t [s]'); ylabel(ax22,'Jarak [m]');
title(ax22,'Jarak USV ke Obs Dinamis','FontWeight','bold');
legend(ax22,'Location','northeast'); grid(ax22,'on');

ax23=subplot(3,1,3,'Parent',tab2);
plot(ax23,t_plot,log.cte(1:Np),'-','Color',c1,'LineWidth',1.8);
hold(ax23,'on');
% Shade LMPC active region
if ~isempty(lmpc_idx)
    d_mode=diff([0;log.mode(1:Np);0]);
    starts_on =find(d_mode==1);
    starts_off=find(d_mode==-1);
    for si=1:length(starts_on)
        t_on =t_plot(min(starts_on(si),Np));
        t_off=t_plot(min(starts_off(si),Np));
        patch(ax23,[t_on t_off t_off t_on],[min(log.cte(1:Np))-0.2 min(log.cte(1:Np))-0.2,...
            max(log.cte(1:Np))+0.2 max(log.cte(1:Np))+0.2],...
            [1 0.85 0.85],'FaceAlpha',0.4,'EdgeColor','none');
    end
end
yline(ax23,0,'k:');
xlabel(ax23,'t [s]'); ylabel(ax23,'CTE [m]');
title(ax23,'Cross-Track Error (merah muda = LMPC aktif)','FontWeight','bold');
grid(ax23,'on');

%% TAB 3: States
tab3=uitab(tg,'Title',' States ');
sTit={'Heading \psi [deg]','Heading Error [deg]','CTE [m]',...
      'Surge u [m/s]','Sway v [m/s]','Yaw Rate r [deg/s]',...
      'Roll \phi [deg]','Roll Rate p [deg/s]','Yaw Torque T_N [Nm]'};
sCol={c1,c3,c4,c1,c2,c3,c4,c1,c2};
sY={rad2deg(log.psi_d(1:Np)),rad2deg(log.psi_e(1:Np)),log.cte(1:Np),...
    log.u(1:Np),log.v(1:Np),rad2deg(log.r(1:Np)),...
    rad2deg(log.phi(1:Np)),rad2deg(log.p(1:Np)),log.TN(1:Np)};
for i=1:9
    axS=subplot(3,3,i,'Parent',tab3);
    plot(axS,t_plot,sY{i},'-','Color',sCol{i},'LineWidth',1.8);
    hold(axS,'on'); yline(axS,0,'k:');
    if i==4, yline(axS,1.5,'r--','LineWidth',1.2); end
    if i==7, yline(axS, rad2deg(deg2rad(30)),'r--'); yline(axS,-rad2deg(deg2rad(30)),'r--'); end
    if i==9, yline(axS,lims.TN,'r--'); yline(axS,-lims.TN,'r--'); end
    for id=1:n_dyn
        if ~isempty(lmpc_idx), xline(axS,t_plot(lmpc_idx(1)),'r:','Alpha',0.4); end
    end
    xlabel(axS,'t [s]'); title(axS,sTit{i},'FontWeight','bold'); grid(axS,'on');
end

%% TAB 4: Collision Snapshots (sama seperti Phase 1 untuk perbandingan)
tab4=uitab(tg,'Title',' Collision Check ');
if isempty(collision_event)
    axOK=axes('Parent',tab4,'Position',[0.1 0.3 0.8 0.4]);
    axis(axOK,'off');
    text(axOK,0.5,0.6,'✓ TIDAK ADA COLLISION','Units','normalized',...
        'HorizontalAlignment','center','FontSize',22,'Color',[0 0.6 0],'FontWeight','bold');
    text(axOK,0.5,0.35,sprintf('LMPC berhasil menghindari %d obs dinamis', n_dyn),...
        'Units','normalized','HorizontalAlignment','center','FontSize',14,'Color',[0 0.5 0]);
    text(axOK,0.5,0.15,sprintf('Total LMPC aktif: %.1fs  |  Max CTE: %.3fm', ...
        lmpc_steps*dt, max(abs(cte_act))),...
        'Units','normalized','HorizontalAlignment','center','FontSize',11,'Color',[0.3 0.3 0.3]);
else
    axErr=axes('Parent',tab4,'Position',[0.1 0.3 0.8 0.4]);
    axis(axErr,'off');
    text(axErr,0.5,0.5,sprintf('%d collision events — perlu tuning LMPC',length(collision_event)),...
        'Units','normalized','HorizontalAlignment','center','FontSize',14,...
        'Color','r','FontWeight','bold');
end

%% TAB 5: Animasi Real-Time
tab5=uitab(tg,'Title',' Animasi Real-Time ');
fprintf('\n=== Menjalankan Animasi Real-Time ===\n');
runAnimation_lmpc(tab5, state, fullPath, smoothPath1, smoothPath2, ...
    obstacles, obs_dyn, log, N_end, dt, xMax, yMax, displayMargin, ...
    goal, start, waypoint, lims, collision_event, ...
    c1, c2, cTrail, cDyn1, cDyn2, theta_c, r_detect, r_safe);

tg.SelectedTab=tab5;
fprintf('=== Phase 2 Selesai. ===\n');

%% ============================================================
% STANDALONE TRAJECTORY PLOT — RRT* + G2CBS + ILOS/PID + LMPC
% Jalankan script ini SETELAH RRTandMPC.m selesai dieksekusi
% (semua variabel workspace harus tersedia)
%% ============================================================

%% ── Validasi variabel workspace ─────────────────────────────
required = {'state','log','fullPath','smoothPath1','smoothPath2',...
            'rawPath1','rawPath2','obstacles','obs_dyn','start',...
            'waypoint','goal','xMax','yMax','displayMargin','Np',...
            'lmpc','r_detect','r_safe','r_clear','dt'};
missing = {};
for i = 1:numel(required)
    if ~exist(required{i},'var'), missing{end+1} = required{i}; end %#ok<AGROW>
end
if ~isempty(missing)
    error('Variabel berikut tidak ditemukan di workspace:\n  %s\nJalankan RRTandMPC.m terlebih dahulu.', ...
        strjoin(missing,', '));
end

%% ── Hitung jarak tempuh & durasi ────────────────────────────
x_log   = state(1:Np, 1);
y_log   = state(1:Np, 2);
t_log   = log.t(1:Np);

dx_step = diff(x_log);
dy_step = diff(y_log);
jarak_tempuh = sum(sqrt(dx_step.^2 + dy_step.^2));
durasi_sim   = t_log(end);
lmpc_steps   = sum(log.mode(1:Np));
cte_act      = log.cte(1:Np);

fprintf('\n=== INFO PLOT TRAJECTORY ===\n');
fprintf('  Jarak tempuh USV : %.2f m\n', jarak_tempuh);
fprintf('  Durasi simulasi  : %.1f s\n',  durasi_sim);
fprintf('  LMPC aktif       : %.1f s (%d steps)\n', lmpc_steps*dt, lmpc_steps);
fprintf('  Max CTE          : %.3f m\n',  max(abs(cte_act)));
fprintf('  RMSE CTE         : %.3f m\n',  sqrt(mean(cte_act.^2)));

%% ── Warna (sama persis dengan RRTandMPC.m) ───────────────────
c1     = [0.00 0.45 0.74];
c2     = [0.85 0.33 0.10];
c3     = [0.47 0.67 0.19];
c4     = [0.49 0.18 0.56];
cObs   = [0.80 0.10 0.10];
cDyn1  = [1.00 0.50 0.00];
cDyn2  = [0.50 0.00 0.80];
cTrail = [0.10 0.75 0.30];

theta_c = linspace(0, 2*pi, 60);
n_dyn   = size(obs_dyn, 1);

%% ═══════════════════════════════════════════════════════════
%  FIGURE 1 — TRAJECTORY UTAMA (identik gaya RRT_Origin)
%% ═══════════════════════════════════════════════════════════
fig1 = figure('Name','Trajectory USV — RRT*+LMPC','Position',[50 50 900 560]);
hold on; grid on; axis equal;
xlim([0 xMax]); ylim([0 yMax]);
xlabel('X [m]', 'FontSize', 11);
ylabel('Y [m]', 'FontSize', 11);
title('Trajektori USV (RRT* + G2CBS + ILOS/PID + LMPC)', ...
    'FontWeight','bold','FontSize',12);

% Obstacles statis
for i = 1:size(obstacles,1)
    xc = obstacles(i,1); yc = obstacles(i,2);
    rc = obstacles(i,3) + displayMargin;
    fill(xc+rc*cos(theta_c), yc+rc*sin(theta_c), cObs, ...
        'FaceAlpha',0.18,'EdgeColor',cObs,'LineWidth',1.4,'HandleVisibility','off');
    fill(xc+obstacles(i,3)*cos(theta_c), yc+obstacles(i,3)*sin(theta_c), cObs, ...
        'FaceAlpha',0.6,'EdgeColor','none','HandleVisibility','off');
    text(xc, yc, sprintf('O%d',i), 'HorizontalAlignment','center', ...
        'FontSize',9,'FontWeight','bold','Color','w');
end

% Obstacles dinamis — posisi awal (▽) & akhir simulasi (lingkaran fill)
t_end_sim = t_log(end);
colors_dyn = {cDyn1, cDyn2};
for id = 1:n_dyn
    oxs = obs_dyn(id,1); oys = obs_dyn(id,2);
    oxe = obs_dyn(id,1) + t_end_sim*obs_dyn(id,4);
    oye = obs_dyn(id,2) + t_end_sim*obs_dyn(id,5);
    % Trail lintasan obs dyn
    ox = log.dyn_pos(1:Np,(id-1)*2+1);
    oy = log.dyn_pos(1:Np,(id-1)*2+2);
    plot(ox, oy, ':', 'Color', colors_dyn{id}, 'LineWidth',1.5, ...
        'DisplayName', sprintf('ObsDyn D%d trail',id));
    % Posisi awal
    plot(oxs, oys, 'v', 'MarkerSize',9, 'MarkerFaceColor',colors_dyn{id}, ...
        'MarkerEdgeColor','k','HandleVisibility','off');
    % Posisi akhir (transparan)
    fill(oxe+obs_dyn(id,3)*cos(theta_c), oye+obs_dyn(id,3)*sin(theta_c), ...
        colors_dyn{id},'FaceAlpha',0.35,'EdgeColor',colors_dyn{id},'HandleVisibility','off');
    text(oxs, oys-0.8, sprintf('D%d',id),'HorizontalAlignment','center', ...
        'FontSize',9,'FontWeight','bold','Color',colors_dyn{id});
end

% Jalur referensi G2CBS (raw RRT* putus-putus)
rawPathAll = [rawPath1; rawPath2(2:end,:)];
plot(rawPathAll(:,1), rawPathAll(:,2), ':', 'Color',[0.6 0.6 0.6], ...
    'LineWidth',1.2,'DisplayName','RRT* Raw');
plot(smoothPath1(:,1), smoothPath1(:,2), '--', 'Color',[c1, 0.7], ...
    'LineWidth',1.8,'DisplayName','G2CBS Seg 1');
plot(smoothPath2(:,1), smoothPath2(:,2), '--', 'Color',[c2, 0.7], ...
    'LineWidth',1.8,'DisplayName','G2CBS Seg 2');

% Trajektori aktual USV
plot(x_log, y_log, '-', 'Color',cTrail, 'LineWidth',2.8, ...
    'DisplayName','Trajektori USV');

% Highlight segmen LMPC aktif
lmpc_idx = find(log.mode(1:Np) == 1);
if ~isempty(lmpc_idx)
    plot(state(lmpc_idx,1), state(lmpc_idx,2), 'o', ...
        'Color',[1 0.2 0.2], 'MarkerSize',3, 'DisplayName','LMPC aktif');
end

% Marker Start / Waypoint / Goal
plot(start(1),    start(2),    '^','MarkerSize',13,'MarkerFaceColor','g', ...
    'MarkerEdgeColor','k','DisplayName','Start');
plot(waypoint(1), waypoint(2), 's','MarkerSize',11,'MarkerFaceColor','b', ...
    'MarkerEdgeColor','k','DisplayName','Waypoint');
plot(goal(1),     goal(2),     'p','MarkerSize',16,'MarkerFaceColor','r', ...
    'MarkerEdgeColor','k','DisplayName','Goal');

% Anotasi statistik
ann_str = sprintf('Jarak: %.2f m | Durasi: %.1f s\nRMSE CTE: %.3f m | LMPC: %.1f s', ...
    jarak_tempuh, durasi_sim, sqrt(mean(cte_act.^2)), lmpc_steps*dt);
annotation('textbox',[0.60 0.02 0.38 0.10],'String',ann_str, ...
    'FontSize',8,'BackgroundColor','w','EdgeColor',[0.7 0.7 0.7],...
    'FitBoxToText','on');

legend('Location','northwest','FontSize',8);

%% ═══════════════════════════════════════════════════════════
%  FIGURE 2 — CTE & MODE TIMELINE (bonus, berguna untuk Bab 4)
%% ═══════════════════════════════════════════════════════════
fig2 = figure('Name','CTE & Mode Timeline — RRT*+LMPC','Position',[980 50 700 500]);

% Subplot 1: Mode switching
subplot(3,1,1);
area(t_log, log.mode(1:Np), 'FaceColor',[1 0.85 0.85],'EdgeColor','none');
hold on;
plot(t_log, log.mode(1:Np), 'r-','LineWidth',1.5);
ylim([-0.1 1.5]);
yticks([0 1]); yticklabels({'ILOS','LMPC'});
xlabel('t [s]'); ylabel('Mode');
title('Mode Switching (0=ILOS, 1=LMPC aktif)','FontWeight','bold');
grid on;

% Subplot 2: Jarak ke obs dinamis
subplot(3,1,2);
colors_dyn2 = {cDyn1, cDyn2};
for id = 1:n_dyn
    plot(t_log, log.dist_dyn(1:Np,id), '-', 'Color',colors_dyn2{id}, ...
        'LineWidth',1.8,'DisplayName',sprintf('Dist D%d',id));
    hold on;
end
yline(r_detect,'b--','LineWidth',1.5,'Label',sprintf('r_{detect}=%.1fm',r_detect));
yline(r_safe,  'r--','LineWidth',1.5,'Label',sprintf('r_{safe}=%.1fm',r_safe));
yline(r_clear, 'g--','LineWidth',1.2,'Label',sprintf('r_{clear}=%.1fm',r_clear));
xlabel('t [s]'); ylabel('Jarak [m]');
title('Jarak USV ke Rintangan Dinamis','FontWeight','bold');
legend('Location','northeast','FontSize',8); grid on;

% Subplot 3: CTE dengan shading LMPC
subplot(3,1,3);
hold on;
% Shade region LMPC aktif
if ~isempty(lmpc_idx)
    d_mode   = diff([0; log.mode(1:Np); 0]);
    starts_on  = find(d_mode ==  1);
    starts_off = find(d_mode == -1);
    y_lim_cte = [min(cte_act)-0.3, max(cte_act)+0.3];
    for si = 1:length(starts_on)
        t_on  = t_log(min(starts_on(si),  Np));
        t_off = t_log(min(starts_off(si), Np));
        patch([t_on t_off t_off t_on], ...
              [y_lim_cte(1) y_lim_cte(1) y_lim_cte(2) y_lim_cte(2)], ...
              [1 0.85 0.85],'FaceAlpha',0.5,'EdgeColor','none','HandleVisibility','off');
    end
end
plot(t_log, cte_act, '-', 'Color',c1,'LineWidth',1.8,'DisplayName','CTE');
yline(0,'k:','HandleVisibility','off');
xlabel('t [s]'); ylabel('CTE [m]');
title('Cross-Track Error (merah muda = LMPC aktif)','FontWeight','bold');
legend('FontSize',8); grid on;

%% ── Export data ke Excel (format seperti teman) ──────────────
T_export = table(t_log, x_log, y_log, ...
    'VariableNames',{'t_log','x_log','y_log'});
T_export.mode = log.mode(1:Np);
T_export.cte  = cte_act;
T_export.dist_D1 = log.dist_dyn(1:Np,1);
T_export.dist_D2 = log.dist_dyn(1:Np,2);

fname = 'trajectory_RRTandMPC.xlsx';
writetable(T_export, fname);
fprintf('\nData diekspor ke: %s\n', fname);
fprintf('  Kolom: t_log | x_log | y_log | mode | cte | dist_D1 | dist_D2\n');
fprintf('  Jarak tempuh USV : %.2f m\n', jarak_tempuh);
fprintf('  Durasi           : %.1f s\n',  durasi_sim);
%% ================================================================
%  LMPC SOLVER
%% ================================================================

function [psi_d_out, U0_out] = lmpc_solve(eta, u_cur, fullPath, wpIdx, ...
    alpha_path, obs_pred, obs_r, r_safe, lmpc, lims)
% ─────────────────────────────────────────────────────────────────
% LMPC Solver menggunakan Quadratic Programming (quadprog)
%
% Formulasi batch LQR:
%   State error: z = [xe, ye, psi_e]  (3×1)
%   Input seq : U = [dp_1;dU_1; dp_2;dU_2; ... dp_N;dU_N]  (2N×1)
%   Prediksi  : Z = M*z0 + C*U
%   Cost      : min U'*H*U + f'*U
%   Subject to: obstacle sebagai soft penalty di H dan f
%
% Ref: Yuan et al. (2022), Gonzalez-Garcia et al. (2022)
% ─────────────────────────────────────────────────────────────────
    N     = lmpc.N;
    dt    = lmpc.dt;
    nWP   = size(fullPath, 1);
    nx    = 3;   % state: [xe, ye, psi_e]
    nu    = 2;   % input: [delta_psi, delta_U]
    r_min = obs_r + r_safe;

    xk = eta(1); yk = eta(2); psik = eta(3);
    U_init = max(lmpc.U_min, min(lmpc.U_max, u_cur));
    U0_nom = lmpc.U0_nom;

    % ── Model linear diskrit ─────────────────────────────────────
    % State error relatif ke path, linearisasi di U0_nom, psi_e≈0
    % xe: along-track error, ye: cross-track error, psi_e: heading error
    %
    % Continuous: dxe/dt = U0  (diabaikan, xe tidak dikontrol ketat)
    %             dye/dt = U0 * psi_e  (small angle)
    %             dpsi_e/dt = (delta_psi)/tau_psi
    % Discrete (Euler, dt):
    A = [1,  0,  0;
         0,  1,  U0_nom*dt;
         0,  0,  1        ];
    B = [0,       0;
         0,       0;
         dt/3.0,  0];   % tau_psi=3s, delta_U tidak masuk state

    % Saat obstacle dekat, kurangi bobot tracking agar obstacle penalty menang
    dist_to_obs = norm([xk - obs_pred(1,1), yk - obs_pred(1,2)]);
    r_min_local = obs_r + r_safe;
    if dist_to_obs < r_min_local * 3.0
        scale = max(0.05, dist_to_obs / (r_min_local * 3.0));
        Q  = lmpc.Q  * scale;   % tracking kurang penting saat dekat obs
        Qf = lmpc.Qf * scale *0.5;
    else
        Q  = lmpc.Q;
        Qf = lmpc.Qf;
    end
    R  = lmpc.R;

    % ── Bangun matriks batch M (nx*N × nx) dan C (nx*N × nu*N) ──
    % Z = M*z0 + C*U
    M = zeros(nx*N, nx);
    C = zeros(nx*N, nu*N);
    Ak = eye(nx);
    for i = 1:N
        Ak = A * Ak;
        M((i-1)*nx+1:i*nx, :) = Ak;
        for j = 1:i
            C((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A^(i-j) * B;
        end
    end

    % ── Bangun Q_bar dan R_bar ───────────────────────────────────
    Q_bar = zeros(nx*N, nx*N);
    for i = 1:N-1
        Q_bar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx) = Q;
    end
    Q_bar((N-1)*nx+1:N*nx, (N-1)*nx+1:N*nx) = Qf;  % terminal

    R_bar = kron(eye(N), R);

    % ── Initial state error ──────────────────────────────────────
    p_ref  = fullPath(min(wpIdx, nWP), :);
    xe0    =  cos(alpha_path)*(xk-p_ref(1)) + sin(alpha_path)*(yk-p_ref(2));
    ye0    = -sin(alpha_path)*(xk-p_ref(1)) + cos(alpha_path)*(yk-p_ref(2));
    psi_e0 = wrapToPi(psik - alpha_path);
    z0     = [xe0; ye0; psi_e0];

    % ── Cost QP dasar (tanpa obstacle) ───────────────────────────
    H_qp = 2*(C'*Q_bar*C + R_bar);
    H_qp = (H_qp + H_qp')/2;  % pastikan simetris
    f_qp = 2*C'*Q_bar*M*z0;

    % ── Obstacle: tentukan sisi menghindar & set hard bound dp ───
    % Estimasi vy obs dari prediksi
    if size(obs_pred,1) >= 2
        vy_obs = (obs_pred(2,2) - obs_pred(1,2)) / dt;
    else
        vy_obs = 0;
    end

    % Posisi obs relatif ke USV dalam frame world
    obs_now_x = obs_pred(1,1);
    obs_now_y = obs_pred(1,2);

    % Cross-track position obs terhadap heading USV
    % Positif = obs di sisi kiri USV, Negatif = obs di sisi kanan
    obs_cross = -sin(psik)*(obs_now_x - xk) + cos(psik)*(obs_now_y - yk);

    % Along-track: seberapa jauh obs di depan USV
    obs_along =  cos(psik)*(obs_now_x - xk) + sin(psik)*(obs_now_y - yk);

    % Hitung clearance yang dibutuhkan
    dist_now = sqrt((xk-obs_now_x)^2 + (yk-obs_now_y)^2);
    clearance_needed = r_min + 0.3;  % sedikit margin ekstra

    % Strategi menghindar berdasarkan POSISI LATERAL obs terhadap USV:
    % Jika obs di kiri USV (obs_cross > 0) → belok KANAN (dp negatif)
    % Jika obs di kanan USV (obs_cross < 0) → belok KIRI (dp positif)
    % Override dengan vy_obs sebagai tiebreaker jika obs tepat di depan
     if vy_obs > 0.1
        % D1: obs naik dari bawah → USV harus tetap di bawah obs → jangan naik
        % Dalam path frame: USV harus ke sisi NEGATIF ye (bawah path)
        avoid_right = true;   % dp negatif = belok kanan = turun dari path
    elseif vy_obs < -0.1
        % D2: obs turun dari atas → USV harus tetap di atas obs → jangan turun  
        avoid_right = false;  % dp positif = belok kiri = naik dari path
    else
        avoid_right = (obs_cross < 0);
    end

    % Hitung dp minimum yang dibutuhkan untuk clear obstacle
    % Berdasarkan geometri: dp ≈ asin(clearance / (U0_nom * time_to_collision))
    time_to_coll = max(0.5, dist_now / max(U0_nom, 0.5));
    lateral_needed = max(0, clearance_needed - abs(obs_cross));
    dp_needed = min(lmpc.dpsi_max, ...
        atan2(lateral_needed, U0_nom * time_to_coll * 0.8));

    % Tambahkan soft penalty ke QP yang mendorong ke arah yang benar
    % Ini jauh lebih reliable dari gradient karena langsung ke komponen dp
    W_obs = 7000;
    dp_idx = 1;  % indeks delta_psi step pertama

    if dist_now < r_min * 4.0
        if avoid_right
            % Dorong dp ke nilai negatif (belok kanan)
            dp_target = -dp_needed;
        else
            % Dorong dp ke nilai positif (belok kiri)
            dp_target = dp_needed;
        end
        % Penalty makin besar makin dekat (kuadratik)
        proximity_factor = (r_min * 4.0 - dist_now) / (r_min * 4.0);
        pen_strength = W_obs * proximity_factor^2 * 3.0;
        f_qp(dp_idx) = f_qp(dp_idx) + 2 * pen_strength * (-dp_target);
        H_qp(dp_idx, dp_idx) = H_qp(dp_idx, dp_idx) + 2 * pen_strength;
    for i_pen = 1:min(3, N)
            pen_idx = (i_pen-1)*2 + 1;
            f_qp(pen_idx) = f_qp(pen_idx) + 2 * pen_strength * 0.45 * (-dp_target);
            H_qp(pen_idx, pen_idx) = H_qp(pen_idx, pen_idx) + pen_strength * 0.5;
        end
    end

    % ── Constraint input ─────────────────────────────────────────

    % ── Constraint input ─────────────────────────────────────────
    % Batas delta_psi dan delta_U untuk semua step
    % Dengan bias: jika obs naik, batasi dpsi ke sisi negatif saja
    if vy_obs > 0.1
        % D1 naik → USV ke bawah → dp negatif → tutup sisi positif
        dp_lb = -lmpc.dpsi_max;
        dp_ub =  deg2rad(3);
    elseif vy_obs < -0.1
        % D2 turun → USV ke atas → dp positif → tutup sisi negatif
        dp_lb = -deg2rad(3);
        dp_ub =  deg2rad(20);
    else
        dp_lb = -lmpc.dpsi_max;
        dp_ub =  lmpc.dpsi_max;
    end
    dU_lb = -0.1;
    dU_ub =  0.1;

    lb = repmat([dp_lb; dU_lb], N, 1);
    ub = repmat([dp_ub; dU_ub], N, 1);

    % ── Solve QP ─────────────────────────────────────────────────
    opts = optimoptions('quadprog', 'Display','off', 'MaxIterations',200);
    U_init_vec = zeros(nu*N, 1);

    try
        [U_opt, ~, exitflag] = quadprog(H_qp, f_qp, [], [], [], [], lb, ub, U_init_vec, opts);
        if exitflag <= 0 || isempty(U_opt)
            % Fallback: solusi unconstrained analitik
            U_opt = -(H_qp + 1e-6*eye(size(H_qp,1))) \ f_qp;
            U_opt = max(lb, min(ub, U_opt));
        end
    catch
        U_opt = zeros(nu*N, 1);
    end

    % ── Ambil step pertama saja (Receding Horizon) ───────────────
    dp_star = U_opt(1);   % delta_psi step pertama
    dU_star = U_opt(2);   % delta_U step pertama

    psi_d_out = wrapToPi(psik + dp_star);
    U0_out    = max(lmpc.U_min, min(lmpc.U_max, U_init + dU_star));
end

%% ================================================================
%  ANIMATION FUNCTION
%% ================================================================

function runAnimation_lmpc(tab, state, fullPath, smoothPath1, smoothPath2, ...
    obstacles, obs_dyn, log, N_end, dt, xMax, yMax, margin, goal, ...
    startPt, wayptPt, lims, collision_event, ...
    c1, c2, cTrail, cDyn1, cDyn2, theta_c, r_detect, r_safe)

n_dyn = size(obs_dyn,1);
colors_dyn={cDyn1,cDyn2};
r_coll_plot = obs_dyn(1,3) + 1.0;

%% ── Axes setup ──────────────────────────────────────────────────
axA=axes('Parent',tab,'Position',[0.03 0.17 0.60 0.79]);
hold(axA,'on'); grid(axA,'on'); axis(axA,'equal');
axA.Color=[1 1 1]; axA.GridColor=[0.2 0.2 0.2]; axA.GridAlpha=0.25;
axA.XColor='k'; axA.YColor='k';
xlim(axA,[0 xMax]); ylim(axA,[0 yMax]);
xlabel(axA,'X [m]','Color','k'); ylabel(axA,'Y [m]','Color','k');
title(axA,'Animasi: LMPC Dynamic Avoidance','Color','k','FontWeight','bold','FontSize',11);

%% ── Obs statis ──────────────────────────────────────────────────
obsC={[1 .3 .3],[1 .5 .2],[1 .7 .2],[.8 .3 1],[.3 .8 1],[.3 1 .5]};
for i=1:size(obstacles,1)
    xc=obstacles(i,1); yc=obstacles(i,2);
    rc=obstacles(i,3)+margin; oc=obsC{mod(i-1,6)+1};
    fill(axA,xc+rc*cos(theta_c),yc+rc*sin(theta_c),oc,'FaceAlpha',.10,'EdgeColor',oc,'LineStyle','--');
    fill(axA,xc+obstacles(i,3)*cos(theta_c),yc+obstacles(i,3)*sin(theta_c),oc,'FaceAlpha',.65,'EdgeColor','none');
    text(axA,xc,yc,sprintf('O%d',i),'Color','k','FontSize',8,...
        'HorizontalAlignment','center','FontWeight','bold');
end

%% ── Path smooth ─────────────────────────────────────────────────
plot(axA,smoothPath1(:,1),smoothPath1(:,2),'-','Color',c1,'LineWidth',2.0,'DisplayName','Path Seg1');
plot(axA,smoothPath2(:,1),smoothPath2(:,2),'-','Color',c2,'LineWidth',2.0,'DisplayName','Path Seg2');

% Marker
plot(axA,startPt(1),startPt(2),'o','MarkerSize',9,'MarkerFaceColor','g','MarkerEdgeColor','k');
plot(axA,wayptPt(1),wayptPt(2),'o','MarkerSize',9,'MarkerFaceColor','b','MarkerEdgeColor','k');
plot(axA,goal(1),goal(2),'o','MarkerSize',9,'MarkerFaceColor','r','MarkerEdgeColor','k');
text(axA,startPt(1)+0.5,startPt(2),'Start','FontSize',8,'Color','k');
text(axA,wayptPt(1)+0.5,wayptPt(2),'WP','FontSize',8,'Color','b');
text(axA,goal(1)+0.5,goal(2),'Goal','FontSize',8,'Color','r');

%% ── Handle dinamis ──────────────────────────────────────────────
% Trail USV — hijau terang, kontras dari path biru & oranye
hTrail = plot(axA,NaN,NaN,'-','Color',cTrail,'LineWidth',2.4);

% Body USV
Ls=1.5; Bs=0.5;
hullX=[.5*Ls,.28*Ls,-.5*Ls,-.5*Ls,.28*Ls];
hullY=[0,.5*Bs,.4*Bs,-.4*Bs,-.5*Bs];
hShip=fill(axA,NaN,NaN,[0.2 0.8 0.3],'FaceAlpha',.9,'EdgeColor','k','LineWidth',1.2);
hBow =plot(axA,NaN,NaN,'->','Color',[0.9 0.8 0],'MarkerSize',8,'LineWidth',2);

% LMPC active indicator pada body
hLmpcRing=plot(axA,NaN,NaN,'o','MarkerSize',22,'Color',[1 0.2 0.2],...
    'LineWidth',2.5,'MarkerFaceColor','none');

% Obs dinamis
hObsBody=gobjects(n_dyn,1); hObsSafe=gobjects(n_dyn,1);
hObsLbl=gobjects(n_dyn,1);  hObsVec=gobjects(n_dyn,1);
for id=1:n_dyn
    hObsBody(id)=fill(axA,NaN,NaN,colors_dyn{id},'FaceAlpha',0.60,'EdgeColor',colors_dyn{id},'LineWidth',1.5);
    hObsSafe(id)=fill(axA,NaN,NaN,colors_dyn{id},'FaceAlpha',0.12,'EdgeColor',colors_dyn{id},'LineStyle','--','LineWidth',1.2);
    hObsLbl(id) =text(axA,NaN,NaN,sprintf('D%d',id),'Color',colors_dyn{id},...
        'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
    hObsVec(id) =quiver(axA,NaN,NaN,NaN,NaN,0,'Color',colors_dyn{id},'LineWidth',2,'MaxHeadSize',0.8);
end

% Mode label
hModeLbl=text(axA,xMax*0.5,yMax*0.97,'','Color','k','FontSize',11,...
    'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','top',...
    'BackgroundColor',[0.95 0.95 0.95 0.8]);
hInfo=text(axA,0.3,yMax*0.97,'','Color','k','FontSize',9,...
    'VerticalAlignment','top','BackgroundColor',[1 1 1 .75]);

%% ── Panel kanan ─────────────────────────────────────────────────
bg=[1 1 1];

axC=axes('Parent',tab,'Position',[0.66 0.68 0.31 0.26]);
axC.Color=bg; axC.XColor='k'; axC.YColor='k';
hold(axC,'on'); grid(axC,'on');
ylabel(axC,'CTE [m]'); title(axC,'Cross-Track Error','Color','k','FontWeight','bold');
hCTE=plot(axC,NaN,NaN,'-','Color',[0 0.4 0.8],'LineWidth',1.5);
yline(axC,0,'--','Color',[.5 .5 .5]);

axSp=axes('Parent',tab,'Position',[0.66 0.40 0.31 0.24]);
axSp.Color=bg; axSp.XColor='k'; axSp.YColor='k';
hold(axSp,'on'); grid(axSp,'on');
ylabel(axSp,'Speed [m/s]'); title(axSp,'USV Speed','Color','k','FontWeight','bold');
hSpd=plot(axSp,NaN,NaN,'-','Color',[0.85 0.33 0.10],'LineWidth',1.5);
yline(axSp,1.5,'--','Color',[0.8 0 0],'Label','U_{max}');

axDst=axes('Parent',tab,'Position',[0.66 0.10 0.31 0.26]);
axDst.Color=bg; axDst.XColor='k'; axDst.YColor='k';
hold(axDst,'on'); grid(axDst,'on');
xlabel(axDst,'t [s]'); ylabel(axDst,'Jarak [m]');
title(axDst,'Jarak ke ObsDyn','Color','k','FontWeight','bold');
hDst=gobjects(n_dyn,1);
for id=1:n_dyn
    hDst(id)=plot(axDst,NaN,NaN,'-','Color',colors_dyn{id},'LineWidth',1.5,...
        'DisplayName',sprintf('D%d',id));
end
yline(axDst,obs_dyn(1,3)+r_safe,'r--','LineWidth',1.5,'Label',sprintf('r_{safe}=%.1fm',obs_dyn(1,3)+r_safe));
yline(axDst,r_detect,'b--','LineWidth',1.2,'Label',sprintf('r_{detect}=%.1fm',r_detect));
legend(axDst,'Location','northeast','FontSize',7);

%% ── Loop animasi ────────────────────────────────────────────────
skip=max(1,round(0.06/dt));
win =round(60/dt);

for k=1:skip:N_end
    if ~ishandle(tab), break; end
    t  =(k-1)*dt;
    xk =state(k,1); yk=state(k,2); psi=state(k,3);
    uk =state(k,5); vk=state(k,6);
    if isnan(xk), break; end

    %% Update obs dyn
    for id=1:n_dyn
        ox=obs_dyn(id,1)+t*obs_dyn(id,4);
        oy=obs_dyn(id,2)+t*obs_dyn(id,5);
        or=obs_dyn(id,3);
        set(hObsBody(id),'XData',ox+or*cos(theta_c),'YData',oy+or*sin(theta_c));
        set(hObsSafe(id),'XData',NaN,'YData',NaN);
        set(hObsLbl(id),'Position',[ox,oy+or+0.5,0]);
        set(hObsVec(id),'XData',ox,'YData',oy,...
            'UData',obs_dyn(id,4)*3,'VData',obs_dyn(id,5)*3);
    end

    %% Update USV
    set(hTrail,'XData',state(1:k,1),'YData',state(1:k,2));
    Rot=[cos(psi) -sin(psi);sin(psi) cos(psi)];
    hw=Rot*[hullX;hullY];
    set(hShip,'XData',xk+hw(1,:),'YData',yk+hw(2,:));
    set(hBow,'XData',[xk,xk+0.55*Ls*cos(psi)],'YData',[yk,yk+0.55*Ls*sin(psi)]);

    %% Mode indicator
    mode_now=log.mode(k);
    if mode_now==1
        set(hLmpcRing,'XData',xk,'YData',yk);
        set(hModeLbl,'String','⚡ LMPC AKTIF — Menghindari Obstacle','Color',[0.8 0 0],...
            'BackgroundColor',[1 0.9 0.9 0.85]);
    else
        set(hLmpcRing,'XData',NaN,'YData',NaN);
        set(hModeLbl,'String','◎ ILOS — Following Global Path','Color',[0 0.5 0],...
            'BackgroundColor',[0.9 1 0.9 0.85]);
    end

    %% Grafik kanan
    kW=max(1,k-win); tW=log.t(kW:k);
    set(hCTE,'XData',tW,'YData',log.cte(kW:k));
    xlim(axC,[tW(1),max(tW(end)+.1,tW(1)+5)]);
    sp=sqrt(log.u(kW:k).^2+log.v(kW:k).^2);
    set(hSpd,'XData',tW,'YData',sp);
    xlim(axSp,[tW(1),max(tW(end)+.1,tW(1)+5)]);
    for id=1:n_dyn
        set(hDst(id),'XData',log.t(kW:k),'YData',log.dist_dyn(kW:k,id));
    end
    xlim(axDst,[tW(1),max(tW(end)+.1,tW(1)+5)]);

    %% Info box
    set(hInfo,'String',sprintf('t=%.1fs\nSpeed=%.2fm/s\nCTE=%.2fm\nD1=%.2fm  D2=%.2fm',...
        t,sqrt(uk^2+vk^2),log.cte(k),log.dist_dyn(k,1),log.dist_dyn(k,2)));

    drawnow limitrate; pause(0.001);
end
fprintf('  Animasi selesai.\n');
end

%% ================================================================
%  LOCAL FUNCTIONS
%% ================================================================
function [Vdot,eta_dot]=usv4dof_python(V,T,psi,phi,P)
    u=V(1);v=V(2);r=V(3);p=V(4);
    u=max(-5,min(5,u));v=max(-3,min(3,v));r=max(-3,min(3,r));p=max(-5,min(5,p));
    phi=max(-deg2rad(30),min(deg2rad(30),phi));
    Fx=T(1);Fy=T(2);Fy_yn=T(3);delta=T(4); K=P.K;
    du=P.A1*v*r+P.A2*u+P.A3*abs(u)*u+P.A4*(abs(u)^2)*u+P.A18*Fx;
    dv=-(1/P.A1)*u*r+P.A5*v+P.A6*abs(v)*v+P.A7*(abs(v)^2)*v+P.A8*abs(r)*v+P.A9*abs(v)*r;
    dp=-K.KpLin*p-K.KpAbs*abs(p)*p-K.KpCub*(abs(p)^2)*p-K.Kphi*sin(phi)+K.Kfy*Fy+K.Kv*v+K.Kr*r+K.Kdelta*delta+K.Kbias;
    dr=-P.A10*v*u+P.A11*u*v+P.A12*r+P.A13*abs(r)*r+P.A14*(abs(r)^2)*r+P.A15*abs(r)*u+P.A16*abs(u)*r+P.A17*abs(u)*u+P.A20*abs(r)*u+P.A21*abs(u)*r+P.A22*abs(u)*u+P.A19*Fy+P.A19*Fy_yn;
    du=max(-10,min(10,du));dv=max(-10,min(10,dv));dr=max(-10,min(10,dr));dp=max(-15,min(15,dp));
    Vdot=[du;dv;dr;dp];
    R=[cos(psi),-sin(psi)*cos(phi),0,0;sin(psi),cos(psi)*cos(phi),0,0;0,0,cos(phi),0;0,0,0,1];
    eta_dot=R*[u;v;r;p];
end

function Ps=smooth_path_g2cbs_c2(P,nPerSeg,~)
    P=remove_dups(P); if size(P,1)<=2,Ps=P;return;end
    N=size(P,1);t=zeros(N,1);
    for i=2:N,t(i)=t(i-1)+norm(P(i,:)-P(i-1,:));end
    if t(end)<1e-9,Ps=P(1,:);return;end
    h=diff(t);Mx=natural_spline_second_derivs(t,P(:,1));My=natural_spline_second_derivs(t,P(:,2));
    mx=zeros(N,1);my=zeros(N,1);sx=diff(P(:,1))./h;sy=diff(P(:,2))./h;
    mx(1)=sx(1)-h(1)*(2*Mx(1)+Mx(2))/6;my(1)=sy(1)-h(1)*(2*My(1)+My(2))/6;
    for i=2:N-1
        mx(i)=0.5*(sx(i-1)+h(i-1)*(Mx(i-1)+2*Mx(i))/6+sx(i)-h(i)*(2*Mx(i)+Mx(i+1))/6);
        my(i)=0.5*(sy(i-1)+h(i-1)*(My(i-1)+2*My(i))/6+sy(i)-h(i)*(2*My(i)+My(i+1))/6);
    end
    mx(N)=sx(end)+h(end)*(Mx(end-1)+2*Mx(end))/6;my(N)=sy(end)+h(end)*(My(end-1)+2*My(end))/6;
    Ps=P(1,:);
    for i=1:N-1
        hi=h(i);b0=P(i,:);b3=P(i+1,:);
        b1=b0+(hi/3)*[mx(i),my(i)];b2=b3-(hi/3)*[mx(i+1),my(i+1)];
        tau=linspace(0,1,nPerSeg)';
        B=(1-tau).^3.*b0+3*(1-tau).^2.*tau.*b1+3*(1-tau).*tau.^2.*b2+tau.^3.*b3;
        if i>1,B=B(2:end,:);end; Ps=[Ps;B];
    end
    Ps=remove_dups(Ps);
end

function M=natural_spline_second_derivs(t,y)
    N=numel(y);h=diff(t); if N<=2,M=zeros(N,1);return;end
    A=zeros(N,N);d=zeros(N,1);A(1,1)=1;A(N,N)=1;
    for i=2:N-1
        A(i,i-1)=h(i-1);A(i,i)=2*(h(i-1)+h(i));A(i,i+1)=h(i);
        d(i)=6*((y(i+1)-y(i))/h(i)-(y(i)-y(i-1))/h(i-1));
    end
    M=A\d;
end

function Q=remove_dups(Q)
    if isempty(Q),return;end
    keep=[true;vecnorm(diff(Q,1,1),2,2)>1e-8];Q=Q(keep,:);
end

function path=repairPathObstacles(path,obs,margin)
    for iter=1:30
        anyFixed=false;
        for i=2:size(path,1)-1
            for j=1:size(obs,1)
                dx=path(i,1)-obs(j,1);dy=path(i,2)-obs(j,2);
                d=sqrt(dx^2+dy^2);excl=obs(j,3)+margin;
                if d<excl
                    if d<1e-6,dx=1;dy=0;d=1;end
                    path(i,1)=obs(j,1)+excl*dx/d;
                    path(i,2)=obs(j,2)+excl*dy/d;anyFixed=true;
                end
            end
        end
        if ~anyFixed,break;end
    end
end

function path=rrtStar(startPt,goalPt,obs,mapSz,rrt,margin)
    xMax=mapSz(2);yMax=mapSz(1);
    nodes=zeros(rrt.maxIter+2,2);parent=zeros(rrt.maxIter+2,1);costArr=zeros(rrt.maxIter+2,1);
    nodes(1,:)=startPt;nNodes=1;foundGoal=false;goalIdx=0;
    for i=1:rrt.maxIter
        if rand<rrt.goalBias,sample=goalPt;else,sample=[rand*xMax,rand*yMax];end
        d=sqrt(sum((nodes(1:nNodes,:)-sample).^2,2));[~,nI]=min(d);
        dir=sample-nodes(nI,:);dL=norm(dir);
        if dL>rrt.stepSize,newPt=nodes(nI,:)+rrt.stepSize*dir/dL;else,newPt=sample;end
        if ~isCF(nodes(nI,:),newPt,obs,margin)||~inBounds(newPt,xMax,yMax),continue;end
        d2=sqrt(sum((nodes(1:nNodes,:)-newPt).^2,2));nIdxs=find(d2<=rrt.rewireRad);
        bP=nI;bC=costArr(nI)+norm(newPt-nodes(nI,:));
        for ni=nIdxs'
            c=costArr(ni)+norm(newPt-nodes(ni,:));
            if c<bC&&isCF(nodes(ni,:),newPt,obs,margin),bC=c;bP=ni;end
        end
        nNodes=nNodes+1;nodes(nNodes,:)=newPt;parent(nNodes)=bP;costArr(nNodes)=bC;
        for ni=nIdxs'
            nc=costArr(nNodes)+norm(nodes(ni,:)-newPt);
            if nc<costArr(ni)&&isCF(newPt,nodes(ni,:),obs,margin),parent(ni)=nNodes;costArr(ni)=nc;end
        end
        if norm(newPt-goalPt)<rrt.goalTol,foundGoal=true;goalIdx=nNodes;end
    end
    if ~foundGoal,d=sqrt(sum((nodes(1:nNodes,:)-goalPt).^2,2));[~,goalIdx]=min(d);end
    path=goalPt;idx=goalIdx;
    while idx~=1,path=[nodes(idx,:);path];idx=parent(idx);end
    path=[startPt;path(2:end,:)];
end

function path=shortcutPath(path,obs,margin,nIter,maxDist)
    for iter=1:nIter
        n=size(path,1);if n<3,break;end
        i=randi(n-2);j=i+1+randi(min(19,n-i-1));
        if j>n,continue;end
        if norm(path(j,:)-path(i,:))<=maxDist&&isCF(path(i,:),path(j,:),obs,margin)
            path=[path(1:i,:);path(j:end,:)];
        end
    end
end

function path=downsamplePath(path,minDist)
    if size(path,1)<3,return;end
    res=path(1,:);
    for i=2:size(path,1)-1
        if norm(path(i,:)-res(end,:))>minDist,res=[res;path(i,:)];end
    end
    path=[res;path(end,:)];
end

function path=chaikinSmooth(path,iterations)
    if size(path,1)<3,return;end
    for iter=1:iterations
        n=size(path,1);smoothed=path(1,:);
        for i=1:n-1
            smoothed=[smoothed;0.75*path(i,:)+0.25*path(i+1,:);0.25*path(i,:)+0.75*path(i+1,:)];
        end
        path=[smoothed;path(end,:)];
    end
end

function f=isCF(p1,p2,obs,margin)
    f=true;
    for s=linspace(0,1,20)
        px=p1(1)+s*(p2(1)-p1(1));py=p1(2)+s*(p2(2)-p1(2));
        for i=1:size(obs,1)
            if sqrt((px-obs(i,1))^2+(py-obs(i,2))^2)<(obs(i,3)+margin),f=false;return;end
        end
    end
end

function b=inBounds(pt,xMax,yMax),b=pt(1)>=0&&pt(1)<=xMax&&pt(2)>=0&&pt(2)<=yMax;end
function s=computeArcLength(path),d=sqrt(sum(diff(path).^2,2));s=[0;cumsum(d)];end

function kappa=computePathCurvature(path)
    n=size(path,1);if n<3,kappa=zeros(n,1);return;end
    x=path(:,1);y=path(:,2);d=sqrt(diff(x).^2+diff(y).^2);d(d<1e-12)=1e-12;s=[0;cumsum(d)];
    xp=zeros(n,1);yp=xp;xpp=xp;ypp=xp;
    for i=2:n-1
        h1=s(i)-s(i-1);h2=s(i+1)-s(i);
        xp(i)=(x(i+1)*h1^2+x(i)*(h2^2-h1^2)-x(i-1)*h2^2)/(h1*h2*(h1+h2));
        yp(i)=(y(i+1)*h1^2+y(i)*(h2^2-h1^2)-y(i-1)*h2^2)/(h1*h2*(h1+h2));
        xpp(i)=2*(x(i+1)*h1-x(i)*(h1+h2)+x(i-1)*h2)/(h1*h2*(h1+h2));
        ypp(i)=2*(y(i+1)*h1-y(i)*(h1+h2)+y(i-1)*h2)/(h1*h2*(h1+h2));
    end
    xp(1)=(x(2)-x(1))/(s(2)-s(1));yp(1)=(y(2)-y(1))/(s(2)-s(1));
    xp(n)=(x(n)-x(n-1))/(s(n)-s(n-1));yp(n)=(y(n)-y(n-1))/(s(n)-s(n-1));
    xpp(1)=xpp(2);ypp(1)=ypp(2);xpp(n)=xpp(n-1);ypp(n)=ypp(n-1);
    den=(xp.^2+yp.^2).^1.5;den(den<1e-9)=1e-9;kappa=(xp.*ypp-yp.*xpp)./den;
end

function psi=computePathHeading(path)
    dx=diff(path(:,1));dy=diff(path(:,2));
    psi=[atan2(dy,dx);atan2(dy(end),dx(end))];
end