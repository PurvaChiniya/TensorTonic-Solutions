select e.name,e.salary, d.dept_name
from employees e 
INNER JOIN departments d 
on e.dept_id = d.id
order by e.name asc