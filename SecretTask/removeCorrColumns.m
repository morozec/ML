function [removingColumns] = removeCorrColumns(X, treshold)
  
  
  m = size(X,1);
  n = size(X,2);  
  
  removingColumns = [];      
  
 
  
  for i=1:n    
    
    if any(removingColumns == i) 
      continue;
    end;
    
    
    for j=i+1:n
      if any(removingColumns == j)
        continue;
      end;
             
      currCorr = corr(X(:,i),X(:,j));
      if (abs(currCorr) > treshold)
        removingColumns = [removingColumns j];
      end
      
    end        
    
  end
    
          
    
  
  
  
  
end
  