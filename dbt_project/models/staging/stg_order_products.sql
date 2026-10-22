select order_id, product_id, add_to_cart_order, reordered, 'prior' as eval_set
from {{ source('lake', 'order_products_prior') }}

union all

select order_id, product_id, add_to_cart_order, reordered, 'train' as eval_set
from {{ source('lake', 'order_products_train') }}
