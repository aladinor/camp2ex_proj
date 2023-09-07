function [error_vector,J] = gamma_moment_inc_fit_nonlin(params, input,bins_diff, actual_output,x)
% N0 > 0; mu > -1; lambda > 0
% params = [N0 mu lambda]; input = bins; actual_output = M_sd; x = fit moments

params(2) = max(-1,params(2)); % gammainc requires mu > -1
params(3) = max(0,params(3));
params(1) = max(0,params(1));

% params = real(params);
params;

N0 = params(1);
mu = params(2);
lambda = params(3);
Dmin = input(1)-bins_diff(1)/2;
Dmax = input(end)+bins_diff(end)/2;

for i = 1:length(x)
    loginc = log( gammainc(lambda*Dmax,1+mu+x(i)) - gammainc(lambda*Dmin,1+mu+x(i)) );
    loggam = gammaln(1+mu+x(i)); % use gammaln to avoid overflow

    fitted_curve(i) = N0 / ( lambda^(1+mu+x(i)) ) * exp(loginc+loggam);
%         ( gammainc(lambda*Dmax,1+mu+x(i)) - gammainc(lambda*Dmin,1+mu+x(i)) )*...
%         gamma(1+mu+x(i));
end
fitted_curve = reshape(fitted_curve,size(actual_output));

% change fitted params to 10^(-99) if they are 0
fitted_curve(find(fitted_curve==0)) = ones(size(find(fitted_curve==0)))*10^(-99);

% error_vector = (log(fitted_curve) - log(actual_output));
error_vector = (fitted_curve - actual_output)./sqrt(actual_output.*fitted_curve);

% load trace.mat ii N0_trace mu_trace lambda_trace;
% ii = ii+1;
% N0_trace(ii) = N0;
% mu_trace(ii) = mu;
% lambda_trace(ii) = lambda;
% save trace.mat ii N0_trace mu_trace lambda_trace;

J = 0;
% if nargout > 1
% 
%     for i = 1:3
%         % %
%         %         inner_int_Dmax = -1 / (mu + x(i) + 1)^2 *( mfun('hypergeom',[mu + x(i) + 1, mu + x(i) + 1],[mu + x(i) + 2, mu + x(i) + 2],-lambda * Dmax)...
%         %             * (lambda * Dmin) ^ (mu + x(i) + 1) + ...
%         %             (mu + x(i) + 1) * ((mu + x(i) + 1) * gammainc(lambda * Dmax,mu + x(i) + 1,'upper') - gamma(mu + x(i) + 2)) * log(lambda * Dmax));
%         %         inner_int_Dmin = -1 / (mu + x(i) + 1)^2 *( mfun('hypergeom',[mu + x(i) + 1, mu + x(i) + 1],[mu + x(i) + 2, mu + x(i) + 2],-lambda * Dmin)...
%         %             * (lambda * Dmin) ^ (mu + x(i) + 1) + ...
%         %             (mu + x(i) + 1) * ((mu + x(i) + 1) * gammainc(lambda * Dmin,mu + x(i) + 1,'upper') - gamma(mu + x(i) + 2)) * log(lambda * Dmin));
%         %
%         %         zero = 1e-30;
%         %         inner_int_zero = -1 / (mu + x(i) + 1)^2 *( mfun('hypergeom',[mu + x(i) + 1, mu + x(i) + 1],[mu + x(i) + 2, mu + x(i) + 2],-lambda * zero)...
%         %             * (lambda * zero) ^ (mu + x(i) + 1) + ...
%         %             (mu + x(i) + 1) * ((mu + x(i) + 1) * gammainc(lambda * zero,mu + x(i) + 1,'upper') - gamma(mu + x(i) + 2)) * log(lambda * zero));
%         %
%         syms b
%         m = mu + x(i);
% 
% %         inner_int_Dmax = double(int(b^m * exp(-b)*log(b),b,1e-10,lambda*Dmax));
% %         inner_int_Dmin = double(int(b^m * exp(-b)*log(b),b,1e-10,lambda*Dmin));
%         
%         
%         dmaxvec = linspace(1e-10,lambda*Dmax,1e6);
%         dminvec = linspace(1e-10,lambda*Dmin,1e6);
%         
%         inner_int_Dmax = sum(dmaxvec.^m.*exp(-dmaxvec) .* log(dmaxvec)*(dmaxvec(2)-dmaxvec(1)));
%         inner_int_Dmin = sum(dminvec.^m.*exp(-dminvec) .* log(dminvec)*(dminvec(2)-dminvec(1)));
%         
%         
%         J(i,1) = (1/(lambda ^ (1 + mu + x(i))) * (gammainc(lambda * Dmax,1+mu+x(i)) - gammainc(lambda*Dmin,1+mu+x(i)))*gamma(1+params(2)+x(i)))/actual_output(i);
% 
%         %         J(i,2) = (N0/(log(lambda) * lambda^(mu + x(i) + 1)) * (inner_int_Dmax - inner_int_Dmin))/actual_output(i);
%         J(i,2) = (-log(lambda) * N0 / lambda^(mu + x(i) + 1) * (gammainc(lambda * Dmax,1+mu+x(i)) - gammainc(lambda * Dmin,1+mu+x(i)))*gamma(1+params(2)+x(i)) + ...
%             N0/lambda^(mu + x(i) + 1) * (inner_int_Dmax - inner_int_Dmin ))/actual_output(i);
%         %
%         %         J(i,2) = -N0/(lambda^(1+mu+x(i)))*(1/(1+mu+x(i))*(lambda*Dmax)^(1/2*mu+1/2*x(i))*exp(-1/2*lambda*Dmax)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmax)-(lambda*Dmin)^(1/2*mu+1/2*x(i))/(1+mu+x(i))*exp(-1/2*lambda*Dmin)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmin))*log(lambda)+N0/(lambda^(1+mu+x(i)))*(-1/(1+mu+x(i))^2*(lambda*Dmax)^(1/2*mu+1/2*x(i))*exp(-1/2*lambda*Dmax)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmax)+1/2/(1+mu+x(i))*(lambda*Dmax)^(1/2*mu+1/2*x(i))*log(lambda*Dmax)*exp(-1/2*lambda*Dmax)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmax)+1/(1+mu+x(i))*(lambda*Dmax)^(1/2*mu+1/2*x(i))*exp(-1/2*lambda*Dmax)*diff(runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmax),mu)-1/2*(lambda*Dmin)^(1/2*mu+1/2*x(i))*log(lambda*Dmin)/(1+mu+x(i))*exp(-1/2*lambda*Dmin)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmin)+(lambda*Dmin)^(1/2*mu+1/2*x(i))/(1+mu+x(i))^2*exp(-1/2*lambda*Dmin)*runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmin)-(lambda*Dmin)^(1/2*mu+1/2*x(i))/(1+mu+x(i))*exp(-1/2*lambda*Dmin)*diff(runWhittakerM(1/2*mu+1/2*x(i),1/2*mu+1/2*x(i)+1/2,lambda*Dmin),mu));
% 
%         J(i,3) = -(N0/(lambda^(mu + x(i) + 2)) * (gammainc(lambda * Dmax,2+mu+x(i)) - gammainc(lambda*Dmin,2+mu+x(i)))*gamma(2+params(2)+x(i)))/actual_output(i);
% 
%         %         J1(i,1) = 1/(lambda ^ (1 + mu + x(i))) * (gammainc(1+mu+x(i),lambda * Dmax) - gammainc(1+mu+x(i),lambda*Dmin));
%         %         J1(i,2) = N0/(log(lambda) * lambda^(mu + x(i) + 1)) * (inner_int_Dmax - inner_int_Dmin);
%         %         J1(i,3) = -N0/(lambda ^ (mu + x(i) +2)) * (gammainc(2+mu+x(i),lambda * Dmax) - gammainc(2+mu+x(i),lambda*Dmin));
%         %
%         %         J(i,1) = (2 * fitted_curve(i) * J1(i,1) - 2 * J1(i,1) * actual_output(i)) / actual_output(i)^2;
%         %         J(i,2) = (2 * fitted_curve(i) * J1(i,2) - 2 * J1(i,2) * actual_output(i)) / actual_output(i)^2;
%         %         J(i,3) = (2 * fitted_curve(i) * J1(i,3) - 2 * J1(i,3) * actual_output(i)) / actual_output(i)^2;
% 
%     end
%     % J = J';
% end