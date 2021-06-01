function [basis,queue]=treeCrossover(X,a,b)
% X ; a: supply; b: destination

tic;

actSet1 = find(a~=0);
actSet2 = find(b~=0);    % 找到�?0项作为积极集

d1 = length(actSet1);
d2 = length(actSet2);    % 积极集的维数

X = X(actSet1,actSet2);  % 去掉X为零的部�?
X = reshape(X, d1, d2);  

aa = sum(X,2);           % 每一行�?�和的列向量
bb = sum(X,1)';          % 每一列�?�和的列向量

tt0 = toc;
t_cross_0 = tt0;

%Yab = zeros(d1,d2);     % Y is the value to be compared. It can be many values.
Yab = X./aa./bb';        % the sum of columns are 1
Ya = X./aa;              % the sum of columns are 1
Yb = X./bb';             % the sum of columns are 1

tt1 = toc;
t_cross_1 = tt1 - tt0;

Yab  = reshape(Yab, [], 1);
Ya = reshape(Ya, [], 1);
Yb  = reshape(Yb, [], 1);
thhold1 = 1.9*mean(Yab(1:97:end));
thhold2 = 1.99*mean(Ya(1:97:end));
thhold3 = 1.99*mean(Yb(1:97:end));
I1 = find(Yab);                       
I2 = find(Ya>thhold2>thhold1);
I3 = find(Yb>thhold3);
[~,II1] = sort(Yab(I1) ,'descend');
[~,II2] = sort(Ya(I2) ,'descend');
[~,II3] = sort(Yb(I3) ,'descend');
I1 = I1(II1);
I2 = I2(II2);
I3 = I3(II3);
minlength = min([length(I1),length(I2),length(I3),int32((d1+d2)/2)])-1;
if minlength >= 1
    I = reshape([I1(1:minlength)';I2(1:minlength)';I3(1:minlength)'],[],1);
    I = [I;I1(minlength+1:end)];
else
    Y = Yab;
    Y = reshape(Y, [], 1);
    [B,I] = sort(Y, 'descend');
end
queue = I;

tt2 = toc;
t_cross_2 = tt2 -tt1;

%

%


%% main function

label = zeros(d1+d2,1);
prenum = int32(min([d1+d2,1e9/(d1+d2)]));
lghSet = zeros(prenum,1);
signSet =  zeros(prenum,1);
signMat = zeros(prenum, d1+d2);
bas = -1 * ones(d1,d2);


counter = 1;   % tree
for iternum = 1:1e6
    ind1 = mod(I(iternum)-1,d1)+1;
    ind2 = int32((I(iternum)-ind1)./d1)+1;
    if label(ind1) == 0 && label(d1 + ind2) == 0
        label(ind1) = counter;
        label(d1+ind2) = counter;
        toInd = counter;
        lghSet(toInd) = 2;
        signSet(toInd) = counter;
        signMat(toInd,1:2) = [ind1, ind2+d1];
    elseif label(d1+ind2) == 0
        tempL = label(ind1);
        label(d1+ind2) = tempL;
        toInd = tempL;
        signMat(toInd,lghSet(toInd)+1) = ind2+d1;
        lghSet(toInd) = lghSet(toInd) + 1;
    elseif label(ind1) == 0
        tempL = label(d1+ind2);
        label(ind1) = tempL;
        toInd = tempL;
        signMat(toInd,lghSet(toInd)+1) = ind1;
        lghSet(toInd) = lghSet(toInd) + 1;
    elseif label(ind1) ~= label(d1+ind2)
        tempL = min([label(ind1),label(d1+ind2)]);
        toInd = tempL;
        fromInd = max([label(ind1),label(d1+ind2)]);
        label(signMat(fromInd,1:lghSet(fromInd))) = tempL;
        signMat(toInd,lghSet(toInd)+1:lghSet(toInd)+lghSet(fromInd)) = signMat(fromInd,1:lghSet(fromInd));
        %signMat(fromInd,1:lghSet(fromInd)) = 0;
        lghSet(toInd) = lghSet(toInd) + lghSet(fromInd);
        lghSet(fromInd) = 0;
        signSet(fromInd) = 0;
    else
        if counter <= d1+d2-1
            continue;
        end
    end
    %         lghSet'
    %     signSet'
    %     signMat
    if counter == d1+d2-1         
        bas(I(iternum)) = 0;
        break
    end
    
    counter = counter + 1;
    bas(I(iternum)) = 0;
end
tmain = toc;
tmain = tmain - tt2 - tt1;
t_cross_main = tmain;

ratio_all2sum = iternum/(d1+d2);
ratio_all2mul = iternum/(d1*d2);


basis = -1*ones(length(a),length(b));
basis(actSet1,actSet2) = bas;
tt3 = toc;
tt3 = tt3 - tmain - tt2 - tt1;
t_cross_3 = tt3;
end