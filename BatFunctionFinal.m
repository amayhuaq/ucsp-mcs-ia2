function BatFunctionFinal( param, textFile, nExp )
    %% se guarda el resultado de 100 experimentos cada uno con 100 iter.
    % en el archivo 'textFile'
    % para optimizar la funcion Ackley con 2d y 10d ejecutar
    % >> paramAckley = [2 1000 100 20 0.5 0 2 -32.768 32.768 1 2];
    % >> BatFunctionFinal(paramAckley, 'Ackley_2d.csv', 100);
    % >> paramAckley = [10 1000 100 20 0.5 0 2 -32.768 32.768 1 2];
    % >> BatFunctionFinal(paramAckley, 'Ackley_10d.csv', 100);
    %
    % para optimizar la funcion Schwefel con 2d y 10d ejecutar
    % >> paramSchwefel = [2 1000 100 100 0.12 0 2 -500 500 1 3];
    % >> BatFunctionFinal(paramSchwefel, 'Schwefel_2d.csv', 100);
    % >> paramSchwefel = [10 1000 100 100 0.12 0 2 -500 500 1 3];
    % >> BatFunctionFinal(paramSchwefel, 'Schwefel_10d.csv', 100);
    %
    % para optimizar la funcion BenchMarking3 con 2d y 10d ejecutar
    % >> paramBench3 = [2 1000 100 0.5 0.5 0 2 -100 100 0 4];
    % >> BatFunctionFinal(paramBench3, 'BenchMarking_2d.csv', 100);
    % >> paramBench3 = [10 1000 100 0.5 0.5 0 2 -100 100 0 4];
    % >> BatFunctionFinal(paramBench3, 'BenchMarking_10d.csv', 100);
    % OJO CONSIDERAR QUE EL A_LOUDNESS Y EL RATIO DEL PULSO SON
    % INICIALIZADOS ALEATORIAMENTE ESTO MEJORA LA PERFORMANCE DEL ALGORITMO
    
    dataCSVEx = zeros(nExp,param(3)/10);    
    for itExp=1:nExp
        dataCSVEx(itExp,:) = BatFunction3(param);
        display(sprintf('iteracion %s %d', textFile, itExp));
    end
    csvwrite(textFile, dataCSVEx);
end

function [fitnessByIter] = BatFunction3( param )

%% INSTANCIAMOS LOS PARAMETROS
D = param(1);
poblaSize = param(2);
numIter = param(3);
% loudness y el ratio de pulso
%A_loudness = param(4)*ones(1,poblaSize);
%r_pulseRatio = param(5)*ones(1,poblaSize);

A_loudness = -1 + 2*rand(1,poblaSize);
r_pulseRatio = rand(1,poblaSize);
r_pulseRatioIni = r_pulseRatio;

r_ini  = param(5);
% para modificar el loudness y el ratio del pulso
alpha = 0.9; %valor obtenido del paper
gamma = 0.9; %valor obtenido del paper
% rango de la frecuentaia
qMin = param(6);
qMax = param(7);
% rango mínimo y máximo de los valores de xi
lower = param(8);
upper = param(9);
%flag maximo o minimo
flag = param(10);
flag2 = param(11);
%array de velocidades
vel = zeros(D, poblaSize);
velPos = zeros(D, poblaSize);
%array de soluciones
x = zeros(D, poblaSize);
x_pos = zeros(D,poblaSize);
fitnessByBat = zeros(1,poblaSize);
fitnessByIter = zeros(1,numIter/10);
bestSol = zeros(D,1);
bestFit = 0;

%% SETEEAMOS LOS PARAMETROS 
for i=1:poblaSize    
    for j=1:D
        randNu = rand(1,1);
        x(j,i) = lower + (upper - lower)*randNu;        
    end
    fitnessByBat(i) = FunObj(x(:,i),flag2);    
end
[bestFit, bestSol] = bestBat(fitnessByBat, x, flag);

%% EMPIEZA EL BUCLE 
itCSV = 1;
for t=1:numIter
    % se genera nuevas soluciones ajustando  las frecuencias
    % actualizando las velocidades y las posiciones            
    
    averLoud = mean(A_loudness);     
    for i=1:poblaSize
        randNu = rand(1,1);
        freqI = qMin + (qMax - qMin)*randNu;        
        velPos(:,i) = vel(:,i) + (x(:,i)-bestSol)*freqI;
        x_pos(:,i) = x(:,i) + velPos(:,i);
        
        randNu = rand(1,1);            
        if randNu > r_pulseRatio(i)
            %display('ENTRA MIERDA');                        
            %rank = (upper - lower)/20;            
            %x_pos(:,i) = bestSol + rank*(-1+2*rand(D,1));            
            randEpsilon = -1 + 2*rand(D,1);
            x_pos(:,i) = bestSol + randEpsilon * averLoud;
            %x_pos(:,i) = bestSol + rank*(-1+2*randn(D,1));                    
        end
        x_pos(:,i) = simpleBounds(x_pos(:,i), lower, upper);                 
        x(:,i) = lower + (upper - lower)*rand(1,D);
                
        %if randNu < A_loudness(i) && FunObj(x_pos(:,i),flag2) < FunObj(x(:,i),flag2)
        if randNu < A_loudness(i) && FunObj(x_pos(:,i),flag2) < FunObj(x(:,i),flag2)
            x(:,i) = x_pos(:,i);
            vel(:,i) = velPos(:,i);
            %A_loudness(i) = alpha * A_loudness(i);
            r_pulseRatio(i) = r_pulseRatioIni(i) * (1 - exp(-1*gamma*t));
        end        
        fitnessByBat(i) = FunObj(x(:,i),flag2);
        [bestFit, bestSol] = bestBat(fitnessByBat, x, flag);        
    end
    if mod(t,10) == 0
        fitnessByIter(1,itCSV) = bestFit;                           
        itCSV = itCSV + 1;
    end    
end
end


%% FUNCIÓN PARA PARAMETRIZAR LOS PARAMETROS
function [val] = simpleBounds(val, low, upp)
    n = length(val);
    for i=1:n
        if val(i) < low
            val(i) = low;
        elseif val(i) > upp
            val(i) = upp;
        end      
    end
end

%% FUNCIÓN EL MEJOR BAT
function [fVal, val] = bestBat(arrayFit, arrayVal, flag)    
    if flag == 1 
        [fVal, pos] = min(arrayFit);
    else
        [fVal, pos] = max(arrayFit);
    end
    val = arrayVal(:,pos);
end

%% FUNCION OBJETIVO
% paramSumSqu = [2 100 100 0.5 0.5 0 2 -2 2 1 1];
% paramSumSqu = [10 100 100 0.5 0.5 0 2 -2 2 1 1];
function [z] = FunObj(x, flag2)    
   if flag2 == 1
       z = sum(x.^2);
   elseif flag2 == 2
       z = FunAckley(x);
   elseif flag2 == 3       
       z = FunSchwefel(x);
   elseif flag2 == 4
       z = FunBenchMarking3(x);
   end
end

%% FUNCTION ACKLEY - BENCHMARKING 1
% xi esta entre [-32.768,32.768]
% minimo global = (0,0, ... ,0) // f = 0
% paramAckley = [2 10000 100 20 0.5 0 2 -32.768 32.768 1 2];
% paramAckley = [10 10000 100 20 0.5 0 2 -32.768 32.768 1 2];
function z=FunAckley(xx)
a = 20;
b = 0.2;
c = 2*pi;
le = length(xx);

sum1 = 0;
sum2 = 0;
for ii=1:le
    xi = xx(ii);
    sum1 = sum1 + xi^2;
    sum2 = sum2 + cos(c*xi);
end

term1 = -a * exp(-b*sqrt(sum1/le));
term2 = -exp(sum2/le);
z = term1 + term2 + a + exp(1);
end

%% FUNCTION SCHWEFEL - BENCHMARKING 2 
% xi esta entre [-500,500]
% minimo global = (420.9687,420.9687, ... , 420.9687) // f = 0
% paramSchwefel = [2 10000 100 100 0.12 0 2 -500 500 1 3];
% paramSchwefel = [10 10000 100 100 0.12 0 2 -500 500 1 3];
function z=FunSchwefel(xx)
     d = length(xx);
     sum = 0;
     for ii=1:d
         xi = xx(ii);
         sum = sum + xi*sin(sqrt(abs(xi)));
     end
     z = 418.9829*d - sum;
end

%% FUNCTION BENCHMARKING3
% xi esta entre [-100,100]
% maximo global = (0,0, ... ,0) // f = 1
% paramBench3 = [2 10000 100 0.5 0.5 0 2 -100 100 0 4];
% paramBench3 = [10 10000 100 0.5 0.5 0 2 -100 100 0 4];
function z=FunBenchMarking3(xx)
    sum1 = sum(xx.^2);
    a = 0.5;
    b = 1.0;
    c = 0.001;
    nume = sin(sqrt(sum1))^2 - a;
    deno = (b+c*sum1)^2;
    z = a - nume/deno;
end



