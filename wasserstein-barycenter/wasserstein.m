
clear

load("C:\2021 Winter\Smart Crossover\Wasserstein Barycenter\mnist-matlab-master\mnist.mat");
width  = test.width;
height = test.height;

eps = 1e-5;
num = [0 1 2 3 4 5 6 7 8 9]';
M   = length(num);
N   = 7;
aph = 1;

d = aph*aph*width*height;

C = zeros(d,d);
for i = 1:d
    for j = 1:d
        x1     = ceil(i/height);
        x2     = ceil(j/height);
        y1     = i - (x1-1)*width;
        y2     = j - (x2-1)*width;
        C(i,j) = (x1-x2)*sign(x1-x2) + (y1-y2)*sign(y1-y2);
    end
end
cc = reshape(C,d^2,1);

A = spalloc(2*N*d+1,N*d^2+d,2*N*d^2+(N+1)*d);
c = zeros(N*d^2+d,1);
for k = 1:N
    A((k-1)*d+1:k*d,(k-1)*d^2+1:k*d^2)         = kron(speye(d),ones(1,d));
    A((N+k-1)*d+1:(N+k)*d,(k-1)*d^2+1:k*d^2  ) = kron(ones(1,d),speye(d));
    A((N+k-1)*d+1:(N+k)*d,N*d^2+1:N*d^2+d)     = -speye(d);
    
    c((k-1)*d^2+1:k*d^2) = cc;
end
A(2*N*d+1,N*d^2+1:N*d^2+d) = ones(1,d);
c(N*d^2+1:N*d^2+d)         = zeros(d,1);

INTCEN      = cell(M,1);
CORCEN      = cell(M,1);
gurCostCros = zeros(M,1);
gurCostBlur = zeros(M,1);
gurTimeCros = zeros(M,1);
gurTimeBlur = zeros(M,1);

for l = 1:M
    
    disp(' ')
    disp(['#####     ',num2str(num(l)),'     #####'])
    disp(' ')
    
    a       = zeros(d,N);
    indset  = find(test.labels==num(l));
    setsize = length(indset);   
    
    for k = 1:N
        ind = randi(setsize);
        aa  = test.images(:,:,ind);
        
        aaa = zeros(aph*width,aph*height);
        for i = 1:width
            for j = 1:height
                aaa(aph*(i-1)+1:aph*i,aph*(j-1)+1:aph*j) = aa(i,j);
            end
        end
        aa     = reshape(aaa,d,1);
        aa     = aa/sum(aa);
        a(:,k) = aa;
    end
    
    b   = [reshape(a,d*N,1); zeros(d*N,1); 1];
    blx = zeros(N*d^2+d,1);
    bux = inf*ones(N*d^2+d,1);
    
    [gurCostCros(l),Y,gurTimeCros(l)]=gurLPBlurCross(A,b,c,blx,bux,eps,'='); 

    [gurCostBlur(l),X,gurTimeBlur(l)]=gurLPBlur(A,b,c,blx,bux,eps,'=');
    
    INTCEN{l} = X{1}(end-d+1:end);
    CORCEN{l} = Y{1}(end-d+1:end);
    
end