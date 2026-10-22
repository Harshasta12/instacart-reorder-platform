select
    p.product_id,
    p.product_name,
    p.aisle,
    p.department,
    count(*)                         as total_orders,
    count(distinct f.user_id)        as unique_buyers,
    avg(f.reordered::double)         as reorder_rate,
    avg(f.add_to_cart_order)         as avg_cart_position
from {{ ref('fct_order_items') }} f
join {{ ref('dim_products') }} p using (product_id)
group by 1, 2, 3, 4
