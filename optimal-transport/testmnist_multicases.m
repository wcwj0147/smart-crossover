% Script to test dual-sinkhorn divergences.
% (c) Marco Cuturi 2013
clear


load("C:\wcwj\Smart Crossover\mnist-matlab-master\mnist.mat");
width = test.width;
height = test.height;

Alpha = [2 2 2 3 3 3 4 4 4 5 5 5]'; 
N = length(Alpha);
eps = 1e-2;

GurobiCostAll = zeros(N,1);
GurobiTimeAll = zeros(N,1);
Gurobicross = zeros(N,1);
Mytime_warm = zeros(N,1);
Mytime_general = zeros(N,1);
time_gap = zeros(N,1);
Suppl = zeros(N,1);
Desti = zeros(N,1);
dimRCost_general = zeros(N,1);
dimRCost_warm = zeros(N,1);
GurobiCostBCross = zeros(N,1);
ratio = zeros(N,1000);
ratio_tree = zeros(N,1);
crosstime = zeros(N,1);
pushtime = zeros(N,1);
dimRTime_warm = zeros(N,1);
dimRTime_general = zeros(N,1);
subcount_warm = zeros(N,1);
subcount1 = zeros(N,1);
subcount2= zeros(N,1);
GurobiTimeAll = zeros(N,1);
GurobiTimeBCross = zeros(N,1);
GurobiTimeBlur = zeros(N,1);


for k = 1:N
    %% Initialization: Randomly Import an Example from Mnist
    
    % pick from the test set
    ind1 = randi(test.count);
    ind2 = randi(test.count);

    % a = reshape(test.images(:,:,ind1),d,1);   
    % b = reshape(test.images(:,:,ind2),d,1);   

    alpha = Alpha(k);  % 图片放大系数
    d = alpha*alpha*width*height;
    a = test.images(:,:,ind1);
    b = test.images(:,:,ind2);
    aa = zeros(alpha*width,alpha*height);
    bb = zeros(alpha*width,alpha*height);
    for i = 1:width
        for j = 1:height
            aa(alpha*(i-1)+1:alpha*i,alpha*(j-1)+1:alpha*j) = a(i,j);
            bb(alpha*(i-1)+1:alpha*i,alpha*(j-1)+1:alpha*j) = b(i,j);
        end
    end
    a = reshape(aa,d,1);    a = a/sum(a);
    b = reshape(bb,d,1);    b = b/sum(b);

    % a = reshape(aa,d,1)+1e-6*rand(d,1);  a = a/sum(a);
    % b = reshape(bb,d,1)+1e-6*rand(d,1);  b = b/sum(b);

    M = zeros(d,d);
    for i = 1:d
        for j = 1:d
            x1 = ceil(i/height);
            x2 = ceil(j/height);
            y1 = i - (x1-1)*width;
            y2 = j - (x2-1)*width;
            M(i,j) = (x1-x2)*sign(x1-x2) + (y1-y2)*sign(y1-y2);
        end
    end

    I = find(a > 0);
    J = find(b > 0);
    a = a(I); 
    b = b(J); 
    M = M(I,J);
    d2 = length(J);
    d1 = length(I);
    Suppl(k) = d2;
    Desti(k) = d1;


    %%  Solve the Example

    disp(' ');
    disp(['         #####  Solve Problem ',num2str(k),'  #####   ']);
    disp(['   ##### Transport from image ', num2str(ind1), ' to image ', num2str(ind2), ' #####' ]);
    disp(['   ##### from the number ', num2str(test.labels(ind2)), ' to the number ', num2str(test.labels(ind1)), ' #####' ]);
    disp(['   ##### amplification factor = ', num2str(alpha), '   #####'])
    disp(['   ##### number of suppliers = ', num2str(d2), ', number of destinations = ', num2str(d1), ' #####']); 
    disp(' ');


    [GurobiCostAll, GurobiTAll, GurobiIterAll, GurobiTimeAll(k), ~] = gurobiTransportAll(a,b,M);
    %[CplexCostAll(k), CplexTAll, CplexTimeAll(k), CplexBasis] = cplexTransportAll(a,b,M);

    [GurobiCostBCross(k), GurobiTBCross, GurobiIterBCross, GurobiTimeBCross(k), GurobiBasis]=gurobiTransportBlurCross(a,b,M,eps); 

    [GurobiCostBlur, GurobiTBlur, GurobiIterBlur, GurobiTimeBlur(k)]=gurobiTransportBlur(a,b,M,eps);

    T = reshape(GurobiTBlur,d1,d2);

    tic;
    [basis,queue] = treeCrossover(T,a,b);
    crosstime(k) = toc;

    tic;
    fbasis = PushPhase3_1(basis,a,b,M);
    pushtime(k) = toc;

    basind = find(GurobiBasis == 0);
    for i = 0:floor(log2((d1*d2)/(d1+d2-1)))
        R = intersect(basind,queue(1:2^i*(d1+d2-1))); 
        ratio(k,i+1) = length(R)/length(basind);
    end
    treeind = find(fbasis == 0);
    R = intersect(treeind,basind);
    ratio_tree(k) = length(R)/length(basind);

    [dimRCost_warm(k),dimROT_warm,dimRTime_warm(k),subcount_warm(k)] = dimReduction_warm(a,b,M,fbasis,queue);

    [dimRCost_general(k),dimROT_general,dimRTime_general(k),subcount1(k),subcount2(k)] = dimReduction_general(a,b,M,queue); 

    Mytime_warm(k) = crosstime(k) + pushtime(k) + dimRTime_warm(k);
    Mytime_general(k) = dimRTime_general(k) + crosstime(k);

    Gurobicross(k) = GurobiTimeBCross(k) - GurobiTimeBlur(k);

    time_gap(k) = dimRTime_warm(k) - dimRTime_general(k);
%     if time_gap(k) < 0
%         disp('     PERFECT!    '); disp(' ');
%     end
    
    
end

err1 = (dimRCost_general-GurobiCostBCross).*sign(dimRCost_general-GurobiCostBCross);
err2 = (dimRCost_warm-GurobiCostBCross).*sign(dimRCost_warm-GurobiCostBCross);
I1 = find(err1>1e-5);
I2 = find(err2>1e-5);
if isempty(I1)&&isempty(I2)
    disp(' ##### Solving Correctly! ##### ')
else
    disp(' ##### Error! #####');
end
disp(' ');


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
