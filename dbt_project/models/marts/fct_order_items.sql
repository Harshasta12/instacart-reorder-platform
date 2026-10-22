select
    op.order_id,
    op.product_id,
    o.user_id,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    o.days_since_prior_order,
    op.add_to_cart_order,
    op.reordered
from {{ ref('stg_order_products') }} op
join {{ ref('stg_orders') }} o using (order_id)
