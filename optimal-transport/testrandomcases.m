% Script to test dual-sinkhorn divergences.
% (c) Marco Cuturi 2013
clear


%% Initialization: 

% Suppl = [1e3 1e3 1e3 1e3 1e3 1e3 1e3 1e3 1e3 1e3];
% Desti = [2e3,4e3,6e3,8e3,10e3,12e3,14e3,16e3,18e3,2e4];

Suppl = [500 1e3 1.5e3 2e3 2.5e3 3e3 3.5e3 4e3 4.5e3 5e3];
Desti = [500 1e3 1.5e3 2e3 2.5e3 3e3 3.5e3 4e3 4.5e3 5e3];

N = length(Desti);
eps = 1e-2;

crosstime = zeros(N,1);
pushtime = zeros(N,1);
Mytime_warm = zeros(N,1);
Mytime_general = zeros(N,1);
Gurobi_cross = zeros(N,1);
Gurobicost = zeros(N,1);
Mycost_warm = zeros(N,1);
Mycost_general = zeros(N,1);
ratio = zeros(N,10);
ratio_tree = zeros(N,1);
time_gap = zeros(N,1);
GurobiTimeAll = zeros(N,1);
GurobiTimeBlur = zeros(N,1);
GurobiTimeBCross = zeros(N,1);

diary 'C:\Smart Crossover\OT problem\gurobi\mylog\log0130_x.txt'
disp('This is log_2 on 01/30.')
disp(' ');
disp(' ');

diary on

%%  Main Program

for k = 1:N
    
    d1 = Desti(k);  d2 = Suppl(k);
    disp(' ');
    disp(['   ##### Solve synthetic problem ', num2str(k), ', d1 = ', num2str(d1), ', d2 = ', num2str(d2), ' #####' ]);
    disp(' ');
    
    %% Generate Synthetic Example
    a=rand(d1,1)+1e-1; a=bsxfun(@rdivide,a,sum(a)); 
    b=rand(d2,1)+1e-1; b=bsxfun(@rdivide,b,sum(b)); 
    M = rand(d1,d2)+1;
    %M = randn(d1,d2)+1;
    
    
    %% Solve the Example
    [GurobiCostAll, GurobiOTAll, GurobiIterAll, GurobiTimeAll(k), ~] = gurobiTransportAll(a,b,M);
   
    [GurobiCostBCross, GurobiTBCross, GurobiIterBCross, GurobiTimeBCross(k), GurobiBasis]=gurobiTransportBlurCross(a,b,M,eps); 

    [GurobiCostBlur, GurobiTBlur, GurobiIterBlur, GurobiTimeBlur(k)]=gurobiTransportBlur(a,b,M,eps);

    disp(' ');
    disp('  ################################# ')
    disp(' ');
    
    T = reshape(GurobiTBlur,d1,d2);
    %T = reshape(CplexTBlur{1},d1,d2);

    tic;
    [basis,queue] = treeCrossover(T,a,b);
    crosstime(k) = toc;

    tic;
    fbasis = PushPhase3_1(basis,a,b,M);
    pushtime(k) = toc;
    
    % 选基与找Feasible Tree的效果分析：
    % basind = find(GurobiBasis == 0);
    basind = find(GurobiBasis == 1);
    for i = 0:floor(log2((d1*d2)/(d1+d2-1)))
        R = intersect(basind,queue(1:2^i*(d1+d2-1))); 
        ratio(k,i+1) = length(R)/length(basind);    
    end
    treeind = find(fbasis == 0);
    R = intersect(treeind,basind);
    ratio_tree(k) = length(R)/length(basind);
    
    
    [dimRCost_warm,dimROT_warm,dimRTime_warm,subcount_warm] = dimReduction_warm(a,b,M,fbasis,queue);

    [dimRCost_general,dimROT_general,dimRTime_general,subcount1,subcount2] = dimReduction_general(a,b,M,queue); 

    Mytime_warm(k) = crosstime(k) + pushtime(k) + dimRTime_warm;
    Mytime_general(k) = dimRTime_general;

    Gurobi_cross(k) = GurobiTimeBCross(k) - GurobiTimeBlur(k);
    
    Mycost_warm(k) = dimRCost_warm;
    Mycost_general(k) = dimRCost_general;
    Gurobicost(k) = GurobiCostBCross;
    
    time_gap(k) = dimRTime_warm - dimRTime_general;     % 衡量tree method的价值, 希望他越小越好
    
end    


%% Examine and Output
err1 = (Mycost_general-Gurobicost).*sign(Mycost_general-Gurobicost);
err2 = (Mycost_warm-Gurobicost).*sign(Mycost_warm-Gurobicost);
I1 = find(err1>1e-6);
I2 = find(err2>1e-6);
if isempty(I1)&&isempty(I2)
    disp(' ##### Solving Correctly! ##### ')
else
    disp(' ##### Error! #####');
end
disp(' ');

Mytime_warm_ave = mean(Mytime_warm)
Mytime_general_ave = mean(Mytime_general)
Gurobi_cross_ave = mean(Gurobi_cross)

diary off

%% Previous Code: Initialization

% rand('seed',111);
% relevant dimensions in this example.
% d1=3; 
% d2=3;
% d1 = 3000;
% d2 = 3000;
% N = 1;
% eps = 1e-2;      % for interior point method using Cplex & Gurobi 
% iterlim = 11;    % for interior point method using Mosek
% 1e-2 --> 10; 1e-3 --> 12; ...

% draw randomly a symmetric cost matrix which is zero on the diagonal. this
% is not a distance matrix, but this suffices to test the script below.
% M = zeros(d1,d2,N);
% for i = 1:N
%     R = rand(d1,d2)+1 ; % 和crossover比, M要大致相同, 和simplex比, M要很不同
%     M(:,:,i) = R/median(R(:)); % normalize to get unit median.
% end

% % set lambda
% lambda=200;
% 
% % the matrix to be scaled.
% K=exp(-lambda*M);
% 
% % in practical situations it might be a good idea to do the following:
% %K(K<1e-100)=1e-100;
% 
% % pre-compute matrix U, the Schur product of K and M.
% U=K.*M;


%% Previous Code: Solving problems 

% disp(' ');
% disp('***** Example when Computing N distances between N different pairs ******');
% draw and normalize 1 point in the simplex with a few zeros (this is not a uniform sampling)
% a=full(sprand(d1,1,1))+1e-1; a=a/sum(a); 

% draw and normalize N points in the simplex with a few zeros (not uniform)
% b=full(sprand(d2,N,1))+1e-1; b=bsxfun(@rdivide,b,sum(b)); 
% a is now updated to be a matrix of column vectors in the simplex.
% a=full(sprand(d1,N,1))+1e-1; a=bsxfun(@rdivide,a,sum(a)); 


%[gurobiCostBCross, gurobiTBCross, gurobiIterBCross, gurobiTimeBCross]=gurobiTransportBlurCross(a,b,M,eps); % running with Gurobi
% [mosekCostBlur, mosekTBlur, mosekTimeBlur]=mosekTransportBlur(a,b,M,iterlim); % running with Gurobi
%[cplexCostBCross, cplexTBCross, cplexTimeBCross]=cplexTransportBlurCross(a,b,M,eps); % running with Gurobi
% [gurobiCostBlur, gurobiTBlur, gurobiIterBlur, gurobiTimeBlur]=gurobiTransportBlur(a,b,M,eps); % running with Gurobi
%[mosekCostBCross, mosekTBCross, mosekTimeBCross]=mosekTransportBlurCross(a,b,M,eps); % running with Gurobi
%[cplexCostBlur, cplexTBlur, cplexTimeBlur]=cplexTransportBlur(a,b,M,eps); % running with Gurobi
%gurobiTimeCross = gurobiTimeBCross - gurobiTimeBlur;
%mosekTimeCross = mosekTimeBCross - mosekTimeBlur;

%cplexTimeCross = sum(cplexTimeBCross - cplexTimeBlur)/N;
%mosekTimeCross = sum(mosekTimeBCross - mosekTimeBlur)/N;

% disp(['Computing ',num2str(N),' distances (a_1,b_1), ... a_',num2str(N),'b_',num2str(N)]);

% t_Sinkhorn = 0;
% tic;
% [D,lowerEMD,l,m]=sinkhornTransport(a,b,K,U,lambda,[],[],[],[],1); % running with VERBOSE
% t_Sinkhorn = toc
% T = zeros(d1,d2,N);
% 
% for i = 1:N
%     T(:,:,i) = diag(l(:,i)) * K * diag(m(:,i));
% end

% [realCost, realOT, timeFinal]=cplexTransportBlur(a,b,M,eps); % running with Gurobi
% T = cell2mat(realOT);
% T = reshape(T,d1,d2);
%[realCost, realOT, iterAll, timeAll]=gurobiTransportAll(a,b,M); % running with Gurobi
%T = cell2mat(realOT);
% verX = zeros(d1,d2,N);
% basis = zeros(d1,d2,N);
% for i = 1:N
%     [verX(:,:,i),basis(:,:,i)] = transportCrossover(T(:,:,i),M,a(:,i),b(:,i),'difference','x/a,x/b');
% end

% T = zeros(d1,d2,N);
% for i = 1:N
%     T(:,:,i) = reshape(cplexTBlur{i},d1,d2);
% end


% tic
% for i = 1:N
%     [basis(:,:,i)] = treeCrossover(T(:,:,i),a(:,i),b(:,i));
% end
% crosstime = toc
% cplexcrosstime = sum(cplexTimeBCross) - sum(cplexTimeBlur)


% [cplexCostFinal, cplexTFInal, cplexTimeFinal]=cplexTransportFinal(a,b,M,basis); 
% MYtime = sum(cplexTimeBlur) + crosstime + sum(cplexTimeFinal)
% MYtime = crosstime + sum(cplexTimeFinal);


%[gurobiCostFinal, gurobiTFInal, gurobiIterFinal, gurobiTimeFinal]=gurobiTransportFinal(a,b,M,verX,basis);




% basis_push3 = zeros(d1,d2,N);
% tic
% for i = 1:N
%     [basis_push3(:,:,i)] = PushPhase3(basis(:,:,i),a(:,i),b(:,i),M(:,:,i));
% end
% pushtime3 = toc

% [cplexCostFinal_push3, cplexTFInal_push3, cplexTimeFinal_push3] = cplexTransportFinal(a,b,M,basis_push3); 
% Mytime = (crosstime + pushtime3 + sum(cplexTimeFinal_push3))/N

% [dimRCost_warm,dimROT_warm,dimRTime_warm,subcount_warm] = dimReduction_warm(a,b,M,basis_push3,T);
% Mytime_old = (crosstime + pushtime3 + sum(dimRTime_warm))/N

% Mytime_old_all = Mytime_old + sum(cplexTimeBlur)/N;

% [mosekCostBCross, mosekTBCross, mosekTimeBCross]=mosekTransportBlurCross(a,b,M,iterlim);
% mosekTimeCross = sum(mosekTimeBCross - mosekTimeBlur)/N
% 
% [cplexCostBCross, cplexTBCross, cplexTimeBCross]=cplexTransportBlurCross(a,b,M,eps); % running with Gurobi
% cplexTimeCross = sum(cplexTimeBCross - cplexTimeBlur)/N
% 
% [gurobiCostBCross, gurobiTBCross, gurobiIterBCross, gurobiTimeBCross]=gurobiTransportBlurCross(a,b,M,eps);
% gurobiTimeCross = sum(gurobiTimeBCross - gurobiTimeBlur)/N

% [dimRCost,dimROT,dimRTime,subcount] = dimReduction(a,b,M,basis_push3,T);
% Mytime_dimR = (crosstime + pushtime3 + sum(dimRTime))/N


%[gurobiCostFinal_push, gurobiTFInal_push, gurobiIterFinal_push, gurobiTimeFinal_push]=gurobiTransportFinal(a,b,M,verX,basis);


% [dimRCost_general,dimROT_general,dimRTime_general,subcount1,subcount2] = dimReduction_general(a,b,M,T); 
% Mytime_new = sum(dimRTime_general)/N

%Mytime_new_all = Mytime_new + sum(cplexTimeBlur)/N;


%[mosekCostAll, mosekTAll, mosekTimeAll]=mosekTransportAll(a,b,M); 
%[cplexCostAll, cplexTAll, cplexTimeAll]=cplexTransportAll(a,b,M); 
%[gurobiCostAll, gurobiTAll, gurobiTimeAll]=gurobiTransportAll(a,b,M); 

% 
% lowerEMD = realCost;
% disp('Done computing distances');
% disp(' ');
% 
% figure()
% cla;
% disp('Display Vector of Distances and Lower Bounds on EMD');
% 
% bar(D,'b');
% hold on
% bar(lowerEMD,'r');
% legend({'Sinkhorn Divergence','Lower bound on EMD'});
% axis tight; title(['Dual-Sinkhorn Divergence and Lower Bound on EMD for ',num2str(N),' pairs (a_i,b_i)'],'FontSize',16); set(gca,'FontSize',16)
% 
% %choose a random pair of histograms a_i, b_i
% %i=round(N*rand());
% i=1;
% disp(['Display (smoothed) optimal transport from a_',num2str(i),' to b_',num2str(i),', which has been chosen randomly.']);
% 
% T=bsxfun(@times,m(:,i)',(bsxfun(@times,l(:,i),K))); % this is the optimal transport. 
% figure()
% imagesc(T);title(['T for (a_{',num2str(i),'},b_{',num2str(i),'})'],'FontSize',16);
% %check that T is indeed a transport matrix.
% disp(['Deviation of T from marginals: ', num2str(norm(sum(T)-b(:,i)')),' ',...
%    num2str(norm(sum(T,2)-a(:,i))),...
%    ' (should be close to zero)']);
% 
% 
% disp(['Display aftercrossover optimal transport from a_',num2str(i),' to b_',num2str(i),', which has been chosen randomly.']);
% T=realOT{i}; % this is the optimal transport. 
% figure()
% imagesc(T);title(['real T for (a_{',num2str(i),'},b_{',num2str(i),'})'],'FontSize',16);
% %check that T is indeed a transport matrix.
% disp(['Deviation of T from marginals: ', num2str(norm(sum(T)-b(:,i)')),' ',...
%    num2str(norm(sum(T,2)-a(:,i))),...
%    ' (should be close to zero)']);
% 
% disp(['Display aftercrossover optimal transport from a_',num2str(i),' to b_',num2str(i),', which has been chosen randomly.']);
% T=verX(:,:,i); % this is the optimal transport. 
% figure()
% imagesc(T);title(['my T for (a_{',num2str(i),'},b_{',num2str(i),'})'],'FontSize',16);
% %check that T is indeed a transport matrix.
% disp(['Deviation of T from marginals: ', num2str(norm(sum(T)-b(:,i)')),' ',...
%    num2str(norm(sum(T,2)-a(:,i))),...
%    ' (should be close to zero)']);
% 
% 
% %disp(['Mosek Computing ',num2str(N),' distances (a_1,b_1), ... a_',num2str(N),'b_',num2str(N)]);
% %[realCost, realOT]=gurobiTransportAll(a,b,M); % running with MOSEK
% %disp('Done computing distances');
% %disp(' ');
