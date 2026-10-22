select
    p.product_id,
    p.product_name,
    p.aisle_id,
    a.aisle,
    p.department_id,
    d.department
from {{ source('lake', 'products') }} p
left join {{ source('lake', 'aisles') }} a using (aisle_id)
left join {{ source('lake', 'departments') }} d using (department_id)
