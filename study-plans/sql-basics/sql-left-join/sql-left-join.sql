select c.name, c.city, COALESCE(SUM(o.amount), 0) AS total_spent
from customers as c 
LEFT JOIN orders as o 
    ON c.id = o.customer_id
GROUP BY c.id, c.name, c.city
ORDER BY total_spent DESC, c.name ASC;