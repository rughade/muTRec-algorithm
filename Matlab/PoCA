function Coord = PoCA_Reshma(p1,p2,p3,p4)


% This function computes the classical PoCA algorithm    

%   Inputs:
%       p1, p2, p3, p4 : 3D coordinates (mm) from top and bottom detector 
                             Pairs
%   Outputs:
%       Coord : Estimated 3D coordinates of PoCA point (mm)
    
    % Direction vectors of the two lines
    u = p2 - p1;  % Direction vector of line incoming line (L1)
    v = p4 - p3;  % Direction vector of line outgoing line (L2)
    
    % Vector w0 = P1 – p3
    w0 = p1 - p3;
 
    % Coefficients for the formula
    a = dot(u, u);        % u·u
    b = dot(u, v);        % u·v
    c = dot(v, v);        % v·v
    d = dot(u, w0);       % u·(p1 - p3)
    e = dot(v, w0);       % v·(p1 - p3)
    
    % Denominator for s_c and t_c
    denom = a * c - b^2;
    
    if denom ~= 0
        % Calculate the parameters s_c and t_c for points of closest approach
        s_c = (b * e - c * d) / denom;  % Parameter for L1
        t_c = (a * e - b * d) / denom;  % Parameter for L2
    else
        % If lines are parallel, set s and t arbitrarily
        s_c = 0;
        t_c = 0;
    end
    
    % Points of closest approach on each line
    P_closest = p1 + s_c * u;  % Point on L1
    Q_closest = p3 + t_c * v;  % Point on L2
    
    % Minimum distance between the two lines
    distance = norm(P_closest - Q_closest);
    
    % Midpoint of the minimum distance
    Coord = (P_closest + Q_closest) / 2;
end
 
 

