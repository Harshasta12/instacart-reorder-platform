-- One row per (user, product): purchase history across prior orders.
with prior as (
    select
        o.user_id,
        op.product_id,
        o.order_id,
        o.order_number,
        op.reordered,
        op.add_to_cart_order
    from {{ ref('stg_order_products') }} op
    join {{ ref('stg_orders') }} o using (order_id)
    where op.eval_set = 'prior'
),

user_orders as (
    select user_id, max(order_number) as max_prior_order_number
    from {{ ref('stg_orders') }}
    where eval_set = 'prior'
    group by 1
)

select
    p.user_id,
    p.product_id,
    count(*)                              as times_bought,
    sum(p.reordered)                      as times_reordered,
    min(p.order_number)                   as first_order_number,
    max(p.order_number)                   as last_order_number,
    avg(p.add_to_cart_order)              as avg_cart_position,
    u.max_prior_order_number - max(p.order_number) as orders_since_last_bought
from prior p
join user_orders u using (user_id)
group by p.user_id, p.product_id, u.max_prior_order_number
