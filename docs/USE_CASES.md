# Tabular ML use cases

A catalog of 59 common tabular machine learning use cases, used by `usecases.py` to turn a goal typed in plain words plus a CSV header into an ML framing: task, target, unit, time handling, metric, leakage suspects and model family priors. The table and per use case notes are generated from `usecases.USE_CASES`, so this file and the code agree. Researched September 2026.

Why it matters for small businesses too: SMB AI adoption rose from 23 percent in 2023 to 58 percent in 2025 [33](https://capsulecrm.com/blog/small-business-ai-adoption-statistics/), and 45 percent of small business owners say they are extremely likely to adopt a tool that predicts revenue trends for staffing, inventory and marketing [34](https://newsroom.paypal-corp.com/2025-06-10-Beyond-Efficiency-Small-Businesses-Look-to-AI-for-Competitive-Edge,-New-Survey-Shows).

## How to read an entry

- **task**: `binary`, `multiclass` or `regression` (the harness maps binary and multiclass to classification).
- **time**: `none` = rows are exchangeable, random stratified CV. `time_split` = rows carry a snapshot or event date; sort by it, use `walk_forward` CV and a latest period holdout, and `purged` CV with gap = label window when label windows overlap (churn in the next 90 days, readmission in 30 days). `forecast` = one target per series per period; `walk_forward` with gap at least the horizon, lag and rolling features shifted by at least the horizon.
- **metric**: one of roc_auc, f1_macro, accuracy (classification) or rmse, mae, r2 (regression). Rules of thumb: roc_auc when people work a ranked list; f1_macro when positives are rare and a single flag drives action (fraud, failures, escalations) or classes are uneven; mae when the target is skewed or zero inflated, because minimizing MAE forecasts the median; rmse when the mean matters (pricing expected loss) or big misses are the costly ones, because minimizing RMSE forecasts the mean [12](https://otexts.com/fpp3/accuracy.html). The user's own words still win: `purpose.rule_spec` and `purpose_spec` override this prior.
- **leakage**: column name patterns for information recorded at or after the outcome. They are suspects for diagnostics to test, not an automatic drop list.
- **families**: multipliers on `search_space.family_prior` (missing family = 1.0).
- **columns**: patterns for the inputs these projects usually have; matching ones raise the match score.

Patterns below are shown without their word boundaries; `*` stands for any word characters.

## Coverage

| segment | use cases |
|---|---|
| small business (`smb`) | 33 |
| mid-market B2B SaaS (`midmarket_saas`) | 29 |
| large enterprise (`enterprise`) | 58 |

| domain | use cases |
|---|---|
| Customer success | 6 |
| Sales and marketing | 10 |
| Finance | 3 |
| Risk and fraud | 7 |
| Operations and supply chain | 9 |
| Pricing | 5 |
| HR | 4 |
| Real estate | 4 |
| Healthcare administration | 5 |
| Manufacturing | 6 |

| task and time | use cases |
|---|---|
| binary, time_split | 33 |
| multiclass, none | 1 |
| multiclass, time_split | 3 |
| regression, forecast | 8 |
| regression, none | 4 |
| regression, time_split | 10 |

## Summary table

| id | use case | segments | task | time | metric | unit |
|---|---|---|---|---|---|---|
| `customer_churn` | Customer churn (subscription or repeat customers) | smb, enterprise | binary | time_split | roc_auc | customer at a snapshot date |
| `saas_renewal_churn` | B2B account renewal and logo churn | midmarket_saas, enterprise | binary | time_split | roc_auc | account per renewal cycle |
| `expansion_upsell` | Expansion and upsell propensity | midmarket_saas, enterprise | binary | time_split | roc_auc | account per quarter |
| `support_escalation` | Support ticket escalation | midmarket_saas, enterprise | binary | time_split | f1_macro | ticket at creation or first response |
| `ticket_resolution_time` | Ticket resolution time | midmarket_saas, enterprise | regression | time_split | mae | ticket |
| `customer_satisfaction` | Customer satisfaction (NPS or CSAT class) | midmarket_saas, enterprise | multiclass | time_split | f1_macro | customer or interaction |
| `lead_scoring` | Lead scoring | smb, midmarket_saas, enterprise | binary | time_split | roc_auc | lead at creation or MQL date |
| `opportunity_win` | Deal (opportunity) win probability | midmarket_saas, enterprise | binary | time_split | roc_auc | opportunity at a pipeline snapshot |
| `campaign_response` | Marketing campaign response | smb, enterprise | binary | time_split | roc_auc | customer per campaign contact |
| `purchase_propensity` | Purchase propensity (will buy in the next N days) | smb, enterprise | binary | time_split | roc_auc | customer at a snapshot date |
| `customer_ltv` | Customer lifetime value | smb, midmarket_saas, enterprise | regression | time_split | mae | customer at a snapshot date |
| `cross_sell` | Cross sell propensity for a product | smb, enterprise | binary | time_split | roc_auc | existing customer |
| `next_best_offer` | Next best offer (which product next) | enterprise | multiclass | time_split | f1_macro | customer at a snapshot date |
| `sales_revenue_forecast` | Sales and revenue forecast (store, region or company) | smb, midmarket_saas, enterprise | regression | forecast | mae | store (or region) per day, week or month |
| `session_conversion` | Website session purchase intent | smb, enterprise | binary | time_split | roc_auc | session |
| `trial_conversion` | Free trial to paid conversion (product qualified leads) | smb, midmarket_saas | binary | time_split | roc_auc | trial account at day N of the trial |
| `invoice_late_payment` | Invoice late payment | smb, midmarket_saas, enterprise | binary | time_split | roc_auc | invoice at issue date |
| `cash_flow_forecast` | Cash flow forecast | smb, midmarket_saas, enterprise | regression | forecast | mae | company or account per week |
| `collections_recovery` | Collections recovery | smb, enterprise | binary | time_split | roc_auc | delinquent account at placement |
| `credit_default` | Credit default risk | smb, enterprise | binary | time_split | roc_auc | loan application or account at origination |
| `transaction_fraud` | Transaction fraud detection | smb, midmarket_saas, enterprise | binary | time_split | f1_macro | transaction |
| `insurance_claim_fraud` | Insurance claim fraud | enterprise | binary | time_split | f1_macro | claim at first notice of loss |
| `aml_alert_scoring` | Anti money laundering alert scoring | enterprise | binary | time_split | roc_auc | alert |
| `application_fraud` | Account opening and application fraud | midmarket_saas, enterprise | binary | time_split | f1_macro | application or signup |
| `claim_probability` | Insurance claim probability (risk per policy) | enterprise | binary | time_split | roc_auc | policy year |
| `claim_severity` | Insurance claim severity (cost of a claim) | enterprise | regression | time_split | mae | claim at first notice of loss |
| `demand_forecast` | Demand forecast (units per product and location) | smb, midmarket_saas, enterprise | regression | forecast | mae | product and location per week (or day) |
| `stockout_risk` | Stockout and backorder risk | smb, enterprise | binary | time_split | f1_macro | product and location per week |
| `late_delivery` | Late delivery risk | smb, midmarket_saas, enterprise | binary | time_split | roc_auc | order or shipment at dispatch |
| `delivery_eta` | Delivery time (ETA) | smb, enterprise | regression | time_split | mae | order or trip |
| `supplier_lead_time` | Supplier lead time | midmarket_saas, enterprise | regression | time_split | mae | purchase order line |
| `call_volume_forecast` | Contact center and staffing volume forecast | midmarket_saas, enterprise | regression | forecast | mae | queue per hour or day |
| `energy_load_forecast` | Energy load forecast | enterprise | regression | forecast | rmse | site or meter per hour |
| `returns_prediction` | Product return prediction | smb, enterprise | binary | time_split | roc_auc | order line |
| `booking_cancellation` | Booking or reservation cancellation | smb, enterprise | binary | time_split | roc_auc | booking at reservation time |
| `employee_attrition` | Employee attrition | smb, midmarket_saas, enterprise | binary | time_split | roc_auc | employee at a snapshot date |
| `offer_acceptance` | Candidate offer acceptance and hiring success | midmarket_saas, enterprise | binary | time_split | roc_auc | candidate or offer |
| `performance_rating` | Employee performance rating | midmarket_saas, enterprise | multiclass | time_split | f1_macro | employee per review cycle |
| `salary_benchmark` | Salary benchmarking | smb, midmarket_saas, enterprise | regression | none | mae | employee or job posting |
| `price_forecast` | Price forecast per product (next period) | smb, midmarket_saas, enterprise | regression | forecast | rmse | product per month (or week) |
| `price_response` | Price response and elasticity (units at a given price) | smb, enterprise | regression | time_split | mae | product per day or week at a price |
| `quote_win` | B2B quote win probability at a price | midmarket_saas, enterprise | binary | time_split | roc_auc | quote |
| `resale_price` | Used vehicle or equipment resale price | smb, enterprise | regression | none | mae | listing or item |
| `insurance_premium` | Insurance premium (expected loss per policy) | enterprise | regression | time_split | rmse | policy year |
| `house_price` | Property value (automated valuation) | smb, enterprise | regression | none | mae | property sale |
| `rent_price` | Rent estimate | smb, enterprise | regression | none | mae | unit or listing |
| `occupancy_forecast` | Occupancy forecast (hotel or property) | smb, enterprise | regression | forecast | mae | property per night or month |
| `lease_renewal` | Tenant lease renewal | smb, enterprise | binary | time_split | roc_auc | lease at N days before expiry |
| `appointment_no_show` | Appointment no show | smb, enterprise | binary | time_split | roc_auc | appointment at booking time |
| `readmission` | Hospital readmission within 30 days | enterprise | binary | time_split | roc_auc | discharge (encounter) |
| `length_of_stay` | Length of stay | enterprise | regression | time_split | mae | admission |
| `claim_denial` | Medical claim denial | midmarket_saas, enterprise | binary | time_split | roc_auc | claim (or claim line) before submission |
| `patient_volume_forecast` | Patient volume forecast | smb, enterprise | regression | forecast | mae | clinic or ward per day (or hour) |
| `predictive_maintenance` | Predictive maintenance (failure in the next N days) | smb, midmarket_saas, enterprise | binary | time_split | f1_macro | machine per day (or cycle) |
| `remaining_useful_life` | Remaining useful life | enterprise | regression | time_split | rmse | asset per cycle |
| `quality_defect` | Quality defect prediction | smb, midmarket_saas, enterprise | binary | time_split | f1_macro | part or batch |
| `yield_prediction` | Process yield | midmarket_saas, enterprise | regression | time_split | rmse | batch or production run |
| `defect_type` | Fault or defect type classification | enterprise | multiclass | none | f1_macro | defect event or part |
| `warranty_claim` | Warranty claim prediction | midmarket_saas, enterprise | binary | time_split | roc_auc | unit sold |

## Notes by domain

### Customer success

Churn is the most repeated tabular use case across vendor libraries: AWS ships churn templates, H2O lists it for banks, payers and telecoms, DataRobot has a churn framing accelerator [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html)[2](https://h2o.ai/solutions/use-case/)[1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html). McKinsey reports telecoms cutting churn by up to 15 percent with analytics [8](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/reducing-churn-in-telecom-through-advanced-analytics). The framing matters more than the model: pick a snapshot date, a label window (say 90 days), and build features only from data before the snapshot [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html). Recency columns such as last_login can encode churn because they stop updating once a customer leaves [22](https://dataconomy.com/2026/09/22/label-leakage-the-bug-that-makes-churn-models-look-brilliant/), so they are kept but checked in diagnostics rather than dropped by name. B2B health scores combine usage, support, relationship and billing signals [35](https://www.accoil.com/blog/customer-health-score).

#### Customer churn (subscription or repeat customers) (`customer_churn`)

- **Question**: Which customers are likely to cancel or stop buying in the next N days? Segments: small business, large enterprise.
- **Phrasings**: "which customers will churn in the next 90 days"; "predict customer churn"; "who is likely to cancel their subscription"; "find customers at risk of leaving"; "flag customers likely to stop buying from us"; "predict attrition of subscribers"; "customer retention risk score"; "probability that a customer churns next month".
- **Framing**: binary; target column like churn, churned, is_churn, churn_flag, churn_label, exited; one row per customer at a snapshot date.
- **Time**: time_split
- **Metric**: roc_auc. Retention teams work a ranked call list, so ranking quality (AUC) matters most; use f1_macro when churners are rare and one flag drives action.
- **Leakage suspects**: `refund*`, `cancel*_(date|reason|at)`, `churn_(date|reason|at)`, `termination*|terminated_*`, `close_date|closed_at|account_closed`, `exit_*|winback*|win_back`, `final_(bill|invoice)|disconnect*`, `retention_offer*|end_date`.
- **Strong families**: catboost x1.2, lightgbm x1.1, logreg x1.1.
- **Columns usually needed**: tenure*, contract*, monthly_charges, support_tickets, last_login*, payment_method.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [8](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/reducing-churn-in-telecom-through-advanced-analytics) [2](https://h2o.ai/solutions/use-case/) [22](https://dataconomy.com/2026/09/22/label-leakage-the-bug-that-makes-churn-models-look-brilliant/).

#### B2B account renewal and logo churn (`saas_renewal_churn`)

- **Question**: Which B2B accounts will not renew at their next renewal date? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which accounts will not renew their contract"; "predict renewal risk for our saas customers"; "customer health score for b2b accounts"; "which logos are likely to churn before renewal"; "forecast gross revenue retention risk by account"; "flag at risk accounts for customer success managers"; "predict whether a subscription renews at contract end".
- **Framing**: binary; target column like renewed, is_renewed, will_renew, renewal, renewal_status, churned; one row per account per renewal cycle.
- **Time**: time_split
- **Metric**: roc_auc. CSMs triage a ranked book of accounts; AUC measures that ranking. Few accounts per period, so expect wide confidence intervals.
- **Leakage suspects**: `churn_(date|reason)|cancel*`, `downgrade*_(date|at)|contraction*`, `renewal_(outcome|result|arr|amount|price)|new_arr|post_renewal*`, `closed_lost*|lost_reason`, `final_invoice|offboard*|exit_*`.
- **Strong families**: catboost x1.2, logreg x1.2, lightgbm x1.1.
- **Columns usually needed**: seats?, arr, dau, tickets?, renewal_date, csm*.
- **Sources**: [35](https://www.accoil.com/blog/customer-health-score) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Expansion and upsell propensity (`expansion_upsell`)

- **Question**: Which existing accounts are ready to buy more seats, a higher plan or an add on? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which accounts are likely to upgrade their plan"; "predict upsell opportunities"; "find customers ready to expand"; "expansion propensity score for existing customers"; "who will buy more seats next quarter"; "predict add on purchases for current accounts".
- **Framing**: binary; target column like expanded, upgraded, upsell, upsold, expansion, is_expansion; one row per account per quarter.
- **Time**: time_split
- **Metric**: roc_auc. Account managers work the top of a ranked list; AUC fits and expansion is a minority class.
- **Leakage suspects**: `new_(arr|mrr|plan|seats)|upgrade_(date|at)|expansion_(amount|arr|date)`, `post_(upgrade|expansion)*`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: seats?, usage, plan, feature*, arr.
- **Sources**: [35](https://www.accoil.com/blog/customer-health-score) [2](https://h2o.ai/solutions/use-case/).

#### Support ticket escalation (`support_escalation`)

- **Question**: Which new support tickets will escalate or breach the SLA? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which tickets will escalate"; "predict sla breaches for support tickets"; "flag tickets likely to need a manager"; "predict which support cases become critical"; "early warning for escalated tickets"; "which cases will be reopened".
- **Framing**: binary; target column like escalated, is_escalated, escalation, sla_breached, breached, reopened; one row per ticket at creation or first response.
- **Time**: time_split
- **Metric**: f1_macro. Escalations are rare and the team acts on a yes or no flag; f1_macro balances missed escalations against false alarms.
- **Leakage suspects**: `resolution_*|resolved_(at|by|date)|closed_(at|date|by)|time_to_resolve*`, `reopen*_count|num_replies|replies|csat*|satisfaction*`, `escalation_(date|at|level|reason)`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: priority, channel, category, tier, first_response*, subject*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Ticket resolution time (`ticket_resolution_time`)

- **Question**: How long will each support ticket take to resolve? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "how long will this ticket take to resolve"; "predict time to resolution for support cases"; "estimate handling time per ticket"; "predict hours until a case is closed"; "forecast resolution time to set customer expectations".
- **Framing**: regression; target column like resolution_hours, resolution_time, time_to_resolve, time_to_resolution, handle_time, hours_to_close; one row per ticket.
- **Time**: time_split
- **Metric**: mae. Durations are right skewed; MAE is robust and targets the median, which is what you promise a customer.
- **Leakage suspects**: `resolved_(at|date)|closed_(at|date)|close_time`, `num_replies|replies|touches|reassign*|csat*|satisfaction*`, `sla_breached|breached`.
- **Strong families**: lightgbm x1.2, catboost x1.1, ridge x0.8.
- **Columns usually needed**: priority, category, channel, created*, team, backlog.
- **Sources**: [2](https://h2o.ai/solutions/use-case/).

#### Customer satisfaction (NPS or CSAT class) (`customer_satisfaction`)

- **Question**: Will this customer be a promoter, passive or detractor? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict nps category for each customer"; "which customers are detractors"; "predict customer satisfaction score"; "will the customer rate us promoter passive or detractor"; "predict csat after a support interaction"; "find unhappy customers before the survey".
- **Framing**: multiclass; target column like nps_category, nps_class, csat, satisfaction, satisfied, rating; one row per customer or interaction.
- **Time**: time_split
- **Metric**: f1_macro. Classes are uneven (detractors are few); f1_macro makes the model care about every class.
- **Leakage suspects**: `survey_(comment|text|response)*|verbatim*|follow_up*|detractor_flag|nps_score`.
- **Strong families**: catboost x1.2, lightgbm x1.1.
- **Columns usually needed**: tickets?, resolution*, tenure, usage, delivery*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [6](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular101).

### Sales and marketing

Lead scoring, propensity to buy, next best offer and cross sell all appear in the H2O marketing and financial services lists [2](https://h2o.ai/solutions/use-case/). Product qualified leads come from in product activation signals (invites, integrations, limits hit) [30](https://productled.com/blog/product-qualified-leads); Einstein opportunity scoring is the CRM default for deal win probability [29](https://www.salesforceben.com/what-is-salesforce-einstein-opportunity-scoring/). The textbook leak is call duration in UCI Bank Marketing: it is only known after the call and must be dropped for a realistic model [18](https://archive.ics.uci.edu/dataset/222/bank+marketing). LTV is zero inflated and heavy tailed; Google's ZILN work shows plain MSE is dominated by top spenders [31](https://arxiv.org/pdf/1912.07753)[7](https://github.com/google-marketing-solutions/crystalvalue).

#### Lead scoring (`lead_scoring`)

- **Question**: Which inbound leads will turn into customers or qualified opportunities? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "score our inbound leads"; "which leads will convert"; "predict lead conversion"; "rank leads by likelihood to buy"; "which prospects should sales call first"; "predict which marketing qualified leads become customers"; "lead scoring model for the crm".
- **Framing**: binary; target column like converted, is_converted, won, is_won, became_customer, sql; one row per lead at creation or MQL date.
- **Time**: time_split
- **Metric**: roc_auc. Reps call leads in score order, so the metric must reward ranking; AUC does, independent of any threshold.
- **Leakage suspects**: `opportunity_(id|amount|stage)|deal_*|close_date|closed_(at|date)`, `converted_(at|date)|won_(at|date)|account_created*|customer_id`, `sales_stage|lifecycle_stage|lead_status`.
- **Strong families**: catboost x1.2, lightgbm x1.1, logreg x1.1.
- **Columns usually needed**: lead_source, industry, page_views, title, country.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [30](https://productled.com/blog/product-qualified-leads).

#### Deal (opportunity) win probability (`opportunity_win`)

- **Question**: Which open deals in the pipeline will close as won? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which deals will close"; "predict opportunity win probability"; "forecast which pipeline deals we will win this quarter"; "score open opportunities in salesforce"; "likelihood that a deal closes won"; "predict win or loss for each sales opportunity".
- **Framing**: binary; target column like won, is_won, closed_won, win, outcome, is_closed_won; one row per opportunity at a pipeline snapshot.
- **Time**: time_split
- **Metric**: roc_auc. Pipeline reviews rank and weight deals; AUC checks the ranking, calibration comes from the confirm table.
- **Leakage suspects**: `close_date|closed_(at|date)|is_closed|days_to_close`, `stage|stage_name|forecast_category|probability`, `lost_reason|loss_reason|won_reason|win_reason`, `final_amount|booked*|contract_signed*`.
- **Strong families**: catboost x1.2, lightgbm x1.1.
- **Columns usually needed**: amount, age_days, industry, product*, owner, meetings?.
- **Sources**: [29](https://www.salesforceben.com/what-is-salesforce-einstein-opportunity-scoring/).

#### Marketing campaign response (`campaign_response`)

- **Question**: Which customers will respond to this campaign, call or offer? Segments: small business, large enterprise.
- **Phrasings**: "who will respond to our marketing campaign"; "predict which customers subscribe after a call"; "target the customers most likely to accept the offer"; "predict email campaign response"; "which contacts will say yes to a term deposit"; "propensity to respond to direct mail".
- **Framing**: binary; target column like y, response, responded, subscribed, accepted, accepted_offer; one row per customer per campaign contact.
- **Time**: time_split
- **Metric**: roc_auc. Budget buys the top k contacts; AUC measures that ordering. Responders are usually under 15 percent.
- **Leakage suspects**: `duration`, `call_duration|talk_time`, `response_(date|at)|redeem*|coupon_used|order_id|order_value`.
- **Strong families**: lightgbm x1.1, catboost x1.1, logreg x1.1.
- **Columns usually needed**: age, channel, campaign*, recency, segment.
- **Sources**: [18](https://archive.ics.uci.edu/dataset/222/bank+marketing) [2](https://h2o.ai/solutions/use-case/).

#### Purchase propensity (will buy in the next N days) (`purchase_propensity`)

- **Question**: Which customers will make another purchase in the next 30 days? Segments: small business, large enterprise.
- **Phrasings**: "which customers will buy again in the next 30 days"; "predict repeat purchase"; "propensity to buy score"; "who is likely to purchase next month"; "predict whether a customer places another order"; "find customers ready to reorder".
- **Framing**: binary; target column like purchased, will_buy, repeat, repeat_purchase, bought, reordered; one row per customer at a snapshot date.
- **Time**: time_split
- **Metric**: roc_auc. Ranks audiences for outreach, so AUC; the label window must start after the feature cutoff.
- **Leakage suspects**: `next_(order|purchase)_(date|amount|id)|days_to_next*|order_count_after*`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: recency, frequency, monetary, sessions?, tenure.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [2](https://h2o.ai/solutions/use-case/) [6](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular101).

#### Customer lifetime value (`customer_ltv`)

- **Question**: How much will each customer spend over the next 12 months? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict customer lifetime value"; "how much will each customer spend next year"; "estimate clv for new customers"; "forecast future revenue per customer"; "which customers will be most valuable"; "predict 12 month spend per customer".
- **Framing**: regression; target column like ltv, clv, cltv, lifetime_value, future_spend, spend_next*; one row per customer at a snapshot date.
- **Time**: time_split
- **Metric**: mae. Spend is zero inflated and heavy tailed; RMSE chases a few whales, MAE stays robust. Use rmse when the planned total (the mean) is what matters.
- **Leakage suspects**: `total_revenue|lifetime_revenue|total_spend|lifetime_spend|orders_after*`.
- **Strong families**: lightgbm x1.2, catboost x1.2, ridge x0.8, mlp x0.8.
- **Columns usually needed**: first_order*, orders?, aov, channel, tenure.
- **Sources**: [7](https://github.com/google-marketing-solutions/crystalvalue) [31](https://arxiv.org/pdf/1912.07753).

#### Cross sell propensity for a product (`cross_sell`)

- **Question**: Which existing customers will buy a second product such as insurance, a card or an add on? Segments: small business, large enterprise.
- **Phrasings**: "which customers will buy our other product"; "cross sell propensity model"; "predict interest in vehicle insurance among health policy holders"; "who will accept a credit card offer"; "predict which customers take a second product"; "find customers to cross sell to".
- **Framing**: binary; target column like response, cross_sell, bought_product, accepted, interested, has_product_*; one row per existing customer.
- **Time**: time_split
- **Metric**: roc_auc. Campaigns target the top of a ranked list; AUC, and positives are usually a minority.
- **Leakage suspects**: `(policy|product|account)_(start|open)*|new_product_id`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: age, products?, tenure, channel, premium.
- **Sources**: [45](https://www.kaggle.com/competitions/playground-series-s4e7) [2](https://h2o.ai/solutions/use-case/).

#### Next best offer (which product next) (`next_best_offer`)

- **Question**: Which product or offer should we show each customer next? Segments: large enterprise.
- **Phrasings**: "which product will the customer buy next"; "next best offer"; "recommend the next product category for each customer"; "which offer should we send each customer"; "predict the next purchase category"; "next best action for each customer".
- **Framing**: multiclass; target column like next_product, next_category, next_offer, next_best_offer, next_purchase_category, offer; one row per customer at a snapshot date.
- **Time**: time_split
- **Metric**: f1_macro. Many offers with uneven popularity; f1_macro stops rare offers from being ignored.
- **Leakage suspects**: `redeem*|offer_accepted_(at|date)|next_order_(id|date|amount)`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: last_product*, recency, age, channel, products?_owned.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Sales and revenue forecast (store, region or company) (`sales_revenue_forecast`)

- **Question**: How much revenue will each store, region or the business make next week, month or quarter? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "forecast monthly revenue"; "predict weekly sales per store"; "revenue forecast for next quarter"; "how much will we sell next month"; "forecast daily sales for each location"; "project sales by region for the budget"; "predict turnover per branch"; "sales forecasting".
- **Framing**: regression; target column like sales, revenue, weekly_sales, daily_sales, monthly_revenue, turnover; one row per store (or region) per day, week or month.
- **Time**: forecast; horizon 1 to 13 weeks (or 1 to 3 months); lags 1, 2, 4 and 52 weeks (7, 14, 28, 364 days if daily); rolling means over 4 and 13 periods; promo, holiday and opening calendars are known ahead and safe.
- **Metric**: mae. Planners compare forecasts to actuals in currency; MAE reads in currency and targets the median. Use rmse when misses on peak days cost disproportionately.
- **Leakage suspects**: `customers|customer_count|transactions|num_transactions|footfall`, `(actual|realized)_*`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, store*, promo*, holiday*, price*.
- **Sources**: [16](https://www.kaggle.com/c/rossmann-store-sales) [5](https://learn.microsoft.com/en-us/samples/azure/machinelearningnotebooks/many-models-solution-accelerator/) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Website session purchase intent (`session_conversion`)

- **Question**: Will this website visit end in a purchase? Segments: small business, large enterprise.
- **Phrasings**: "will this visitor buy"; "predict which website sessions convert"; "online shopper purchase intention"; "predict checkout conversion from browsing behaviour"; "which visitors are likely to purchase today"; "predict cart abandonment".
- **Framing**: binary; target column like revenue, purchased, converted, conversion, bought, ordered; one row per session.
- **Time**: time_split
- **Metric**: roc_auc. Offers go to the most likely buyers, so ranking (AUC); roughly 15 percent of sessions convert.
- **Leakage suspects**: `order_(id|value|total)|transaction_(id|value)|checkout_completed|payment*|thank_you*`.
- **Strong families**: lightgbm x1.2, xgboost x1.1.
- **Columns usually needed**: page_?views, duration*, bounce*, traffic*, device.
- **Sources**: [46](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

#### Free trial to paid conversion (product qualified leads) (`trial_conversion`)

- **Question**: Which trial users or workspaces will become paying customers? Segments: small business, mid-market B2B SaaS.
- **Phrasings**: "which trial users will convert to paid"; "predict free to paid conversion"; "score product qualified leads"; "find trials most likely to upgrade"; "which signups become paying customers"; "predict freemium upgrade".
- **Framing**: binary; target column like converted, is_converted, paid, is_paid, paying, is_paying; one row per trial account at day N of the trial.
- **Time**: time_split
- **Metric**: roc_auc. Sales assist reaches out to the top trials; ranking (AUC) matters and positives are few.
- **Leakage suspects**: `subscription_(start|id|date)|first_payment*|payment_(date|id)`, `plan_price|mrr|arr|invoice*`, `converted_(at|date)|upgrade_(date|at)`, `card_on_file|billing_(start|id)`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: signup*, sessions?, invites?, integrations?, activated.
- **Sources**: [30](https://productled.com/blog/product-qualified-leads).

### Finance

Finance teams predict whether and when an invoice gets paid, then roll that into a cash forecast and a collections call list [27](https://arxiv.org/pdf/1912.10828)[38](https://arxiv.org/html/2008.07363v1). Everything recorded at or after payment (payment date, amount paid, dunning level, write off) leaks.

#### Invoice late payment (`invoice_late_payment`)

- **Question**: Which open invoices will be paid late? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which invoices will be paid late"; "predict late payment on invoices"; "which customers will pay after the due date"; "prioritize collections by risk of late payment"; "flag invoices at risk of going overdue"; "accounts receivable late payment risk".
- **Framing**: binary; target column like late, is_late, paid_late, overdue, is_overdue, late_payment; one row per invoice at issue date.
- **Time**: time_split
- **Metric**: roc_auc. Collectors call in order of risk, so AUC. For the number of days late, use a regression target with mae.
- **Leakage suspects**: `pay(ment)?_date|paid_(at|date|on)|clear*_date|days_to_pay|days_late|days_overdue`, `amount_paid|paid_amount|outstanding_after*`, `collection_status|dunning*|write_?off*`.
- **Strong families**: lightgbm x1.2, catboost x1.2.
- **Columns usually needed**: invoice_amount, terms?, due_date, customer*, avg_days_late, industry.
- **Sources**: [27](https://arxiv.org/pdf/1912.10828) [38](https://arxiv.org/html/2008.07363v1).

#### Cash flow forecast (`cash_flow_forecast`)

- **Question**: How much cash will come in and go out each week for the next quarter? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "forecast our cash flow"; "predict weekly cash balance"; "how much cash will we have next month"; "forecast receivables collections by week"; "cash position forecast for treasury"; "predict net cash flow per week"; "forecast monthly expenses per cost center".
- **Framing**: regression; target column like net_cash_flow, cash_flow, cashflow, cash_balance, balance, inflow; one row per company or account per week.
- **Time**: forecast; horizon 1 to 13 weeks; lags 1, 4, 13 and 52 weeks; receivables and payables scheduled to fall due inside the horizon are known ahead and safe.
- **Metric**: mae. Treasury reads errors in currency; MAE is interpretable and less driven by one lumpy payment.
- **Leakage suspects**: `(actual|realized|reconciled)_*|closing_balance|ending_balance`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, receivables?, payables?, payroll, revenue.
- **Sources**: [38](https://arxiv.org/html/2008.07363v1) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html) [2](https://h2o.ai/solutions/use-case/).

#### Collections recovery (`collections_recovery`)

- **Question**: Which delinquent accounts will pay if we contact them? Segments: small business, large enterprise.
- **Phrasings**: "which overdue accounts will pay"; "predict collection success"; "prioritize debt collection calls"; "probability of recovering a past due balance"; "which delinquent customers will cure"; "score accounts in collections".
- **Framing**: binary; target column like paid, recovered, cured, collected, promise_kept, is_recovered; one row per delinquent account at placement.
- **Time**: time_split
- **Metric**: roc_auc. Agents work accounts in score order; AUC measures that ranking.
- **Leakage suspects**: `amount_(recovered|collected)|recovered_amount|payment_received*|settle*|write_?off*|closed_(at|date)`.
- **Strong families**: lightgbm x1.2, logreg x1.1.
- **Columns usually needed**: balance, days_past_due, contact*, promise*, history*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/).

### Risk and fraud

AWS ships fraud (XGBoost with SMOTE) and credit decision (LightGBM plus SHAP) templates [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html); H2O lists transaction, application, check and collusion fraud and AML [2](https://h2o.ai/solutions/use-case/); DataRobot has an AML alert scoring accelerator [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html). Lending Club style extracts carry post origination fields (recoveries, total_rec_prncp, last payment) that push a naive AUC to 0.9999 against about 0.70 with origination data only [20](https://github.com/alyrraza/pit-scorecard). Insurance splits into claim probability, scored by normalized Gini (a ranking metric) [15](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction), and claim severity, scored by MAE [14](https://www.kaggle.com/competitions/allstate-claims-severity/overview/evaluation).

#### Credit default risk (`credit_default`)

- **Question**: Will this borrower default on the loan or card? Segments: small business, large enterprise.
- **Phrasings**: "predict loan default"; "which applicants will not repay"; "credit risk score for loan applications"; "probability of default for each borrower"; "who will miss payments on their loan"; "predict charge offs"; "score credit applications".
- **Framing**: binary; target column like default, is_default, defaulted, bad, bad_loan, charged_off; one row per loan application or account at origination.
- **Time**: time_split
- **Metric**: roc_auc. Credit is judged on ranking (AUC, Gini = 2 AUC minus 1) before a cutoff is set; defaults are often under 10 percent.
- **Leakage suspects**: `loan_status`, `recover*|collection*`, `last_pymnt*|last_payment*|total_rec_*|total_pymnt*|out_prncp*|next_pymnt*`, `charge_?off_date|settlement*|debt_settlement*|hardship*`.
- **Strong families**: logreg x1.3, lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: income, loan_amnt, dti, credit_score, term, emp_length, purpose.
- **Sources**: [13](https://www.kaggle.com/competitions/home-credit-default-risk) [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [20](https://github.com/alyrraza/pit-scorecard) [2](https://h2o.ai/solutions/use-case/).

#### Transaction fraud detection (`transaction_fraud`)

- **Question**: Is this payment, order or transaction fraudulent? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "flag fraud"; "flag fraudulent transactions"; "detect fraud in card payments"; "which transactions are fraudulent"; "predict chargebacks"; "fraud detection model for payments"; "catch fraudulent orders before shipping".
- **Framing**: binary; target column like is_fraud, isfraud, fraud, fraudulent, is_fraudulent, fraud_flag; one row per transaction.
- **Time**: time_split
- **Metric**: f1_macro. Fraud is often under 1 percent; AUC can look great while the alert queue is mostly false alarms. f1_macro scores the actual flag; check precision at the chosen threshold.
- **Leakage suspects**: `chargeback_(date|amount|reason)|dispute*`, `investigat*|reviewed_by|review_(outcome|decision)|case_id|analyst*`, `fraud_(reported|type|reason)|reported_fraud`, `blocked|account_blocked|card_blocked|refund*|reversal*`.
- **Strong families**: lightgbm x1.3, xgboost x1.2, catboost x1.1, mlp x0.8, tabicl x0.7.
- **Columns usually needed**: amount, merchant*, time, device*, country, velocity*, card*.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [17](https://www.kaggle.com/competitions/playground-series-s3e4) [2](https://h2o.ai/solutions/use-case/).

#### Insurance claim fraud (`insurance_claim_fraud`)

- **Question**: Is this insurance claim fraudulent and worth a special investigation? Segments: large enterprise.
- **Phrasings**: "detect fraudulent insurance claims"; "which claims should go to the special investigations unit"; "predict claim fraud"; "flag suspicious auto insurance claims"; "score claims for fraud risk"; "which policies are fraudulent".
- **Framing**: binary; target column like fraud_reported, is_fraud, fraud, fraudulent, fraud_flag, fraud*; one row per claim at first notice of loss.
- **Time**: time_split
- **Metric**: f1_macro. Investigators can open few cases and fraud is rare; f1_macro scores the referral flag.
- **Leakage suspects**: `siu_*|investigat*|referral_(outcome|result)`, `claim_(status|denied|decision)|denied|paid_amount|payout*|settle*`, `closed_(at|date)`.
- **Strong families**: lightgbm x1.2, catboost x1.2, xgboost x1.1.
- **Columns usually needed**: claim_amount, policy_*, incident_*, witnesses, insured_*, deductable.
- **Sources**: [47](https://www.kaggle.com/competitions/fraud-detection-in-insurance-claims) [2](https://h2o.ai/solutions/use-case/).

#### Anti money laundering alert scoring (`aml_alert_scoring`)

- **Question**: Which AML alerts will turn into a suspicious activity report? Segments: large enterprise.
- **Phrasings**: "score anti money laundering alerts"; "which aml alerts are true positives"; "predict suspicious activity reports"; "reduce false positives in transaction monitoring"; "prioritize aml alerts for investigators"; "detect money laundering".
- **Framing**: binary; target column like sar, sar_filed, is_sar, suspicious, true_positive, productive; one row per alert.
- **Time**: time_split
- **Metric**: roc_auc. Investigators work alerts in score order; AUC measures that ranking and supports a documented cutoff.
- **Leakage suspects**: `sar_(date|id|filed_date)|case_(closed|outcome|disposition)*|disposition*`, `investigator*|analyst*|notes?|narrative*`.
- **Strong families**: lightgbm x1.2, xgboost x1.1, logreg x1.1.
- **Columns usually needed**: amount*, count*, country*, customer_type, alert_type.
- **Sources**: [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html) [2](https://h2o.ai/solutions/use-case/).

#### Account opening and application fraud (`application_fraud`)

- **Question**: Is this new account, signup or credit application fake or fraudulent? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "detect fake account signups"; "predict application fraud"; "flag fraudulent loan applications"; "identify synthetic identities at onboarding"; "which new accounts are fraudsters"; "detect bank account opening fraud".
- **Framing**: binary; target column like fraud_bool, is_fraud, fraud, fraudulent, fake, is_fake; one row per application or signup.
- **Time**: time_split
- **Metric**: f1_macro. Fraud is rare and onboarding needs an accept or review decision; f1_macro scores that flag.
- **Leakage suspects**: `account_(closed|blocked|banned)*|banned|suspended*`, `first_payment_default|fpd|chargeback*`, `review_(outcome|decision)|manual_review_result|kyc_(result|decision)`.
- **Strong families**: lightgbm x1.3, xgboost x1.2, catboost x1.1.
- **Columns usually needed**: email*, phone*, device*, ip*, income, velocity*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/).

#### Insurance claim probability (risk per policy) (`claim_probability`)

- **Question**: Will this policyholder file a claim in the next year? Segments: large enterprise.
- **Phrasings**: "predict whether a driver files a claim next year"; "probability of an insurance claim per policy"; "which policyholders will make a claim"; "risk score for auto insurance customers"; "predict claim frequency for underwriting"; "safe driver prediction".
- **Framing**: binary; target column like claim, target, has_claim, claimed, claim_flag, is_claim; one row per policy year.
- **Time**: time_split
- **Metric**: roc_auc. Underwriting uses the normalized Gini (2 AUC minus 1), a ranking metric.
- **Leakage suspects**: `claim_(amount|cost|paid|date)|loss*|paid*|incurred*|reserve*`.
- **Strong families**: lightgbm x1.2, xgboost x1.1, catboost x1.1.
- **Columns usually needed**: age, vehicle*, region, prior_claims?, coverage*.
- **Sources**: [15](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

#### Insurance claim severity (cost of a claim) (`claim_severity`)

- **Question**: How much will this claim end up costing? Segments: large enterprise.
- **Phrasings**: "predict the cost of each insurance claim"; "estimate claim severity"; "how much will this claim pay out"; "set initial reserves for new claims"; "predict loss amount per claim"; "claims cost prediction".
- **Framing**: regression; target column like loss, claim_amount, claim_cost, severity, paid_amount, incurred; one row per claim at first notice of loss.
- **Time**: time_split
- **Metric**: mae. Claim costs are heavy tailed; MAE (the Allstate benchmark metric) keeps a few huge claims from dominating.
- **Leakage suspects**: `settle*|final_reserve|reserve_final|closed_(at|date)|litigation_outcome|paid_to_date`.
- **Strong families**: lightgbm x1.2, catboost x1.2, xgboost x1.1.
- **Columns usually needed**: claim_type, injury*, policy*, vehicle*, region, report_lag.
- **Sources**: [14](https://www.kaggle.com/competitions/allstate-claims-severity/overview/evaluation) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

### Operations and supply chain

Demand forecasting is the lead predictive use case in the Azure accelerators and AWS templates [4](https://msusazureaccelerators.github.io/accelerators/demand-forecasting.html)[3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html), and Gartner's supply chain prism lists 21 AI use cases across plan, source, make and deliver [10](https://www.gartner.com/en/documents/3999399). In M5, pooled LightGBM models on lag and rolling features dominated, and cross learning across series beat one model per series [11](https://www.sciencedirect.com/science/article/pii/S0169207021001874). Same period outcomes are the leak: Rossmann's Customers column does not exist at prediction time [16](https://www.kaggle.com/c/rossmann-store-sales); hotel reservation_status and its date encode the cancellation [21](https://github.com/manarsabryqassem32-web/End-to-end-Hotel-Booking-Cancellation-Prediction).

#### Demand forecast (units per product and location) (`demand_forecast`)

- **Question**: How many units of each product will each store or warehouse sell next week? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "forecast weekly demand per store"; "predict units sold per product next week"; "demand forecasting for inventory planning"; "how many items will we sell per sku"; "forecast daily demand by store and item"; "predict product demand for replenishment"; "forecast order quantities per warehouse".
- **Framing**: regression; target column like units, units_sold, quantity, qty, demand, sales_qty; one row per product and location per week (or day).
- **Time**: forecast; horizon 1 to 4 weeks or 1 to 28 days (M5 used 28 days); lags 7, 14, 28 days or 1, 2, 4, 52 weeks; rolling mean and std over 7 and 28; planned price and promo are known ahead. Pool all series in one model.
- **Metric**: mae. Demand is intermittent with many zeros; MAE targets the median and reads in units. Use rmse when stockouts on peaks are the costly error.
- **Leakage suspects**: `lost_sales|stockout*|stock_out*`, `(end|closing)_(stock|inventory)*|fill_rate|fulfilled*`, `returns?_qty|returned_units`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, store*, item*, price*, promo*, holiday*.
- **Sources**: [11](https://www.sciencedirect.com/science/article/pii/S0169207021001874) [4](https://msusazureaccelerators.github.io/accelerators/demand-forecasting.html) [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [10](https://www.gartner.com/en/documents/3999399).

#### Stockout and backorder risk (`stockout_risk`)

- **Question**: Which products will run out of stock before the next delivery? Segments: small business, large enterprise.
- **Phrasings**: "which products will go out of stock"; "predict stockouts next week"; "flag items at risk of running out"; "backorder prediction"; "predict out of stock risk per store"; "which skus will be backordered".
- **Framing**: binary; target column like stockout, is_stockout, out_of_stock, oos, went_on_backorder, backorder; one row per product and location per week.
- **Time**: time_split
- **Metric**: f1_macro. Stockouts are rare and planners act on a flag; f1_macro weighs misses against false alarms.
- **Leakage suspects**: `lost_sales|backorder_qty|unfilled*|end_(stock|inventory)|closing_(stock|inventory)`.
- **Strong families**: lightgbm x1.2, xgboost x1.2.
- **Columns usually needed**: on_hand, lead_time, forecast*, in_transit*, reorder*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [10](https://www.gartner.com/en/documents/3999399).

#### Late delivery risk (`late_delivery`)

- **Question**: Which orders or shipments will arrive late? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which orders will be delivered late"; "predict late deliveries"; "flag shipments at risk of delay"; "on time delivery prediction"; "will this shipment miss its promised date"; "predict delivery delays by carrier".
- **Framing**: binary; target column like late, is_late, late_delivery, late_delivery_risk, delayed, is_delayed; one row per order or shipment at dispatch.
- **Time**: time_split
- **Metric**: roc_auc. Ops teams rank shipments to intervene; AUC. Late rates are often high, so ranking is the need.
- **Leakage suspects**: `actual_*|delivered_(at|date)|delivery_date|arrival_(date|time)|days_for_shipping_real|delivery_status|delay_days|days_late`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: shipping_mode, scheduled*, carrier*, origin*, weight*, order_date.
- **Sources**: [28](https://www.sciencedirect.com/science/article/pii/S294986352300002X) [10](https://www.gartner.com/en/documents/3999399).

#### Delivery time (ETA) (`delivery_eta`)

- **Question**: How long will this delivery or trip take? Segments: small business, large enterprise.
- **Phrasings**: "predict delivery time"; "estimate eta for each order"; "how long will the trip take"; "predict transit time in days"; "estimate food delivery duration"; "predict shipping time from warehouse to customer".
- **Framing**: regression; target column like delivery_time, eta, transit_days, transit_time, duration, trip_duration; one row per order or trip.
- **Time**: time_split
- **Metric**: mae. Customers feel absolute minutes late; MAE reads in minutes and ignores rare stuck orders.
- **Leakage suspects**: `delivered_(at|time|date)|arrival*|dropoff*|actual_*|end_time|completed_at`.
- **Strong families**: lightgbm x1.3, xgboost x1.1.
- **Columns usually needed**: distance*, pickup*, carrier, weather*, region.
- **Sources**: [10](https://www.gartner.com/en/documents/3999399).

#### Supplier lead time (`supplier_lead_time`)

- **Question**: How many days will this purchase order take to arrive from the supplier? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict supplier lead times"; "how long until this purchase order arrives"; "which suppliers will deliver late"; "estimate procurement lead time per order"; "predict days from order to receipt"; "supplier delivery reliability forecast".
- **Framing**: regression; target column like lead_time, lead_time_days, days_to_receive, actual_lead_time, delivery_days, lead_time*; one row per purchase order line.
- **Time**: time_split
- **Metric**: mae. Planners set safety stock in days; MAE in days is readable and robust to a few very late POs.
- **Leakage suspects**: `receipt_date|received_(at|date|qty)|goods_receipt*|gr_date|invoice_date`.
- **Strong families**: catboost x1.2, lightgbm x1.2.
- **Columns usually needed**: supplier*, item*, order_qty, order_date, promised*, country.
- **Sources**: [42](https://zenodo.org/records/17838266) [28](https://www.sciencedirect.com/science/article/pii/S294986352300002X).

#### Contact center and staffing volume forecast (`call_volume_forecast`)

- **Question**: How many calls, chats or tickets will arrive per interval, so we can staff? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "forecast call volume per hour"; "how many support tickets will we get next week"; "predict contact center volume for staffing"; "forecast daily chats and calls"; "workforce planning forecast for agents"; "predict the number of orders per hour for staffing".
- **Framing**: regression; target column like calls, call_volume, volume, contacts, tickets, tickets_created; one row per queue per hour or day.
- **Time**: forecast; horizon next day to 6 weeks at 15 minute to daily intervals; lags at the same interval 1 and 7 days back and 52 weeks back; holiday, campaign and release calendars.
- **Metric**: mae. Staffing reads the error as people per interval; MAE is readable and robust to outage spikes.
- **Leakage suspects**: `handled*|answered*|abandon*|service_level|asa|aht`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, day_of_week, holiday*, campaign*, queue.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [5](https://learn.microsoft.com/en-us/samples/azure/machinelearningnotebooks/many-models-solution-accelerator/).

#### Energy load forecast (`energy_load_forecast`)

- **Question**: How much electricity or gas will each site use per hour tomorrow? Segments: large enterprise.
- **Phrasings**: "forecast electricity demand per hour"; "predict energy consumption for each building"; "load forecasting for tomorrow"; "how much power will we use next week"; "forecast gas usage per site"; "predict kwh consumption per meter".
- **Framing**: regression; target column like load, consumption, kwh, energy, demand, mw; one row per site or meter per hour.
- **Time**: forecast; day ahead (24 to 48 hours); lags 24, 48 and 168 hours; use the weather forecast for the target hour, not the observed weather, or the backtest leaks.
- **Metric**: rmse. Grid and capacity costs are driven by peaks; RMSE penalizes big misses at peak hours.
- **Leakage suspects**: `(actual|metered)_*|bill*_amount`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: datetime, temp*, holiday*, building*, square_feet.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html).

#### Product return prediction (`returns_prediction`)

- **Question**: Will this order or item be returned? Segments: small business, large enterprise.
- **Phrasings**: "which orders will be returned"; "predict product returns"; "flag purchases likely to be sent back"; "return probability per order line"; "predict returns for fashion orders"; "which customers return items often".
- **Framing**: binary; target column like returned, is_returned, return, return_flag, was_returned, return*; one row per order line.
- **Time**: time_split
- **Metric**: roc_auc. Interventions (size advice, review) go to the riskiest orders; AUC measures that ranking.
- **Leakage suspects**: `return_(date|reason|id|status)|refund*|restock*|rma*|credit_note*`.
- **Strong families**: catboost x1.2, lightgbm x1.2.
- **Columns usually needed**: category, price, size*, customer_return*, channel.
- **Sources**: [49](https://www.sciencedirect.com/science/article/abs/pii/S0969698926001086) [50](https://link.springer.com/article/10.1007/s42979-025-03944-z).

#### Booking or reservation cancellation (`booking_cancellation`)

- **Question**: Will this hotel, restaurant or service booking be cancelled? Segments: small business, large enterprise.
- **Phrasings**: "which bookings will be cancelled"; "predict hotel reservation cancellations"; "will the guest cancel"; "forecast cancellation risk for reservations"; "flag bookings likely to cancel for overbooking"; "predict order cancellations before shipping".
- **Framing**: binary; target column like is_canceled, is_cancelled, canceled, cancelled, cancellation, booking_status; one row per booking at reservation time.
- **Time**: time_split
- **Metric**: roc_auc. Overbooking and reminders act on the riskiest bookings; AUC. Cancel rates are often 25 percent or more.
- **Leakage suspects**: `reservation_status*|cancel*_(date|at|reason)|refund*|assigned_room*|check_?in_(time|at)|checked_in`.
- **Strong families**: lightgbm x1.2, catboost x1.2, xgboost x1.1.
- **Columns usually needed**: lead_time, deposit*, adr, market_segment, previous_cancellations, arrival*, special_requests.
- **Sources**: [21](https://github.com/manarsabryqassem32-web/End-to-end-Hotel-Booking-Cancellation-Prediction).

### Pricing

AWS's price optimization template estimates elasticity with double ML and Prophet [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html); B2B quote pricing learns win and loss patterns by discount and deal size [9](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution)[36](https://www.wipro.com/analytics/machine-learning-for-b2b-pricing/). The factory predicts outcomes at a price; it does not identify causal elasticity, so price effects are associations unless prices were varied on purpose.

#### Price forecast per product (next period) (`price_forecast`)

- **Question**: What will the price of each product (or input) be next month? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict next month's price per product"; "forecast future prices for each item"; "what will the price be next month"; "forecast commodity and supplier prices"; "predict the market price of our products next quarter"; "price forecast per product over time".
- **Framing**: regression; target column like price, unit_price, avg_price, average_price, sell_price, selling_price; one row per product per month (or week).
- **Time**: forecast; horizon 1 to 3 months; lags 1, 2, 3 and 12 months; rolling means over 3 and 12; input cost indices lagged by at least the horizon.
- **Metric**: rmse. Prices move in levels and a few large misses break margins; RMSE punishes them and targets the mean. Use mae if outlier months should not dominate.
- **Leakage suspects**: `actual_*|realized_*|settle*_price`.
- **Strong families**: lightgbm x1.3, xgboost x1.1, catboost x1.1, ridge x1.1, random_forest x0.7, tabicl x0.6.
- **Columns usually needed**: date, product*, cost*, competitor*, demand, promo*.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [9](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution).

#### Price response and elasticity (units at a given price) (`price_response`)

- **Question**: How many units will we sell at a given price, so we can pick the best price? Segments: small business, large enterprise.
- **Phrasings**: "what happens to sales if we raise prices"; "estimate price elasticity per product"; "find the optimal price for each product"; "how does demand change with price"; "dynamic pricing model"; "predict units sold at different price points"; "markdown optimization for clearance".
- **Framing**: regression; target column like units, units_sold, quantity, qty, demand, sales; one row per product per day or week at a price.
- **Time**: time_split
- **Metric**: mae. The model predicts units at a price; MAE in units is readable. Elasticity read off it is an association unless prices varied independently of demand.
- **Leakage suspects**: `revenue|sales_amount|turnover|gross_sales`.
- **Strong families**: lightgbm x1.2, xgboost x1.1, ridge x1.1.
- **Columns usually needed**: price*, discount*, competitor*, date, product*.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [2](https://h2o.ai/solutions/use-case/) [9](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution).

#### B2B quote win probability at a price (`quote_win`)

- **Question**: Will this quote be accepted at the price we are offering? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "will the customer accept this quote"; "predict quote win rate at a given discount"; "probability a price quote converts to an order"; "how much discount do we need to win the deal"; "predict whether a bid wins"; "quote acceptance model for pricing".
- **Framing**: binary; target column like won, accepted, is_won, quote_won, converted, awarded; one row per quote.
- **Time**: time_split
- **Metric**: roc_auc. Win curves by price need well ranked probabilities; AUC, then read calibration.
- **Leakage suspects**: `order_(id|date|number)|po_number|po_*|invoice*`, `final_(price|discount|amount)|booked*|closed_(at|date)`.
- **Strong families**: catboost x1.2, lightgbm x1.1, logreg x1.1.
- **Columns usually needed**: quoted_price, discount*, customer*, product*, quantity, competitor*.
- **Sources**: [36](https://www.wipro.com/analytics/machine-learning-for-b2b-pricing/) [9](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution).

#### Used vehicle or equipment resale price (`resale_price`)

- **Question**: What is this used car, machine or item worth on the resale market? Segments: small business, large enterprise.
- **Phrasings**: "predict used car prices"; "estimate resale value of equipment"; "how much is this vehicle worth"; "price used items for our listings"; "predict the selling price of second hand products"; "trade in value estimate".
- **Framing**: regression; target column like price, sale_price, selling_price, sold_price, resale_value, value; one row per listing or item.
- **Time**: none
- **Metric**: mae. Buyers and sellers think in currency per item; MAE is readable and robust to rare luxury listings.
- **Leakage suspects**: `price_per_*|final_price|hammer_price|days_to_sell|sold_(at|date)`.
- **Strong families**: catboost x1.3, lightgbm x1.1.
- **Columns usually needed**: make, model*, year, mileage, condition*, location.
- **Sources**: none dedicated; completes the catalog.

#### Insurance premium (expected loss per policy) (`insurance_premium`)

- **Question**: What premium should we charge this policy for its expected losses? Segments: large enterprise.
- **Phrasings**: "set insurance premiums from risk"; "predict expected loss per policy"; "estimate the premium for each customer"; "insurance pricing model"; "predict annual insurance cost per policyholder"; "price health insurance by customer profile".
- **Framing**: regression; target column like premium, pure_premium, expected_loss, charges, loss_cost, annual_premium; one row per policy year.
- **Time**: time_split
- **Metric**: rmse. Pricing needs the expected (mean) loss; RMSE targets the mean, while MAE would predict near zero for the many policies without claims.
- **Leakage suspects**: `claim_(amount|paid|count)*|paid_losses|incurred*|loss_ratio`.
- **Strong families**: lightgbm x1.2, xgboost x1.1.
- **Columns usually needed**: age, bmi, vehicle*, region, coverage*, exposure.
- **Sources**: [2](https://h2o.ai/solutions/use-case/) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

### HR

IBM's synthetic HR attrition table (1,470 rows) is the standard benchmark; with SMOTE kept inside the CV folds LightGBM reaches ROC AUC about 0.79 [41](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)[23](https://www.mdpi.com/2078-2489/17/3/308). Termination fields and exit interviews are the leaks. HR users must explain decisions, so linear models get a prior boost.

#### Employee attrition (`employee_attrition`)

- **Question**: Which employees are likely to leave in the next year? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "which employees will quit"; "predict employee attrition"; "who is at risk of leaving the company"; "staff turnover prediction"; "flight risk score for employees"; "predict resignations in the next 12 months".
- **Framing**: binary; target column like attrition, left, quit, resigned, terminated, turnover; one row per employee at a snapshot date.
- **Time**: time_split
- **Metric**: roc_auc. HR ranks people for stay interviews; AUC. Leavers are a minority and HR needs to explain the model, so linear models get a boost.
- **Leakage suspects**: `termination*|term_date|separation*|exit_*|last_(day|working_day)|date_of_termination|dateoftermination`, `rehire*|severance*|final_pay*|notice_(date|period_start)|employment_status|emp_status`.
- **Strong families**: logreg x1.3, catboost x1.1, lightgbm x1.1.
- **Columns usually needed**: tenure, salary, job_level, department, overtime, satisfaction*, promotion*.
- **Sources**: [41](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) [23](https://www.mdpi.com/2078-2489/17/3/308) [2](https://h2o.ai/solutions/use-case/).

#### Candidate offer acceptance and hiring success (`offer_acceptance`)

- **Question**: Will this candidate accept our offer, or get through the hiring process? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "will the candidate accept the job offer"; "predict offer acceptance"; "which applicants will be hired"; "predict which candidates drop out of the hiring funnel"; "candidate success prediction"; "which recruits will join after the offer".
- **Framing**: binary; target column like accepted, offer_accepted, hired, joined, is_hired, accept*; one row per candidate or offer.
- **Time**: time_split
- **Metric**: roc_auc. Recruiters prioritise candidates by likelihood; AUC measures that ranking.
- **Leakage suspects**: `start_date|join(ing)?_date|onboard*|employee_(id|number)|hire_date|decline_reason`.
- **Strong families**: catboost x1.2, logreg x1.2.
- **Columns usually needed**: source, offered_*, expected_*, notice_period, role, location.
- **Sources**: [51](https://github.com/kousiksiva/Job-Acceptance-Prediction-System) [52](https://www.noon.ai/blog/articles/21-predictive-hiring-analytics-guide).

#### Employee performance rating (`performance_rating`)

- **Question**: What performance rating will each employee get this cycle? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict employee performance ratings"; "which employees will be high performers"; "predict review scores for staff"; "forecast performance category per employee"; "identify future top performers"; "classify employees into performance bands".
- **Framing**: multiclass; target column like performance_rating, rating, performance_score, perf_rating, performance, performance_band; one row per employee per review cycle.
- **Time**: time_split
- **Metric**: f1_macro. Top and bottom bands are small; f1_macro keeps them from being ignored.
- **Leakage suspects**: `bonus*|raise*|merit*|salary_increase*|calibrated_*|final_rating`.
- **Strong families**: catboost x1.2, lightgbm x1.1, logreg x1.1.
- **Columns usually needed**: tenure, training*, previous_*rating, department, kpis?, manager*.
- **Sources**: [2](https://h2o.ai/solutions/use-case/).

#### Salary benchmarking (`salary_benchmark`)

- **Question**: What is a fair salary for this role, level and location? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict salary for a job"; "what should we pay for this role"; "estimate market compensation by title and location"; "salary benchmarking model"; "predict pay for employees to find underpaid staff"; "estimate base salary from experience".
- **Framing**: regression; target column like salary, base_salary, salary_in_usd, salary_usd, compensation, pay; one row per employee or job posting.
- **Time**: none
- **Metric**: mae. People reason about pay gaps in currency; MAE is readable and robust to executive outliers.
- **Leakage suspects**: `bonus*|total_comp*|tax*|net_pay|salary_currency_amount`.
- **Strong families**: catboost x1.3, lightgbm x1.1.
- **Columns usually needed**: job_title, level, years_*, location, company_size, education*.
- **Sources**: none dedicated; completes the catalog.

### Real estate

Automated valuation moved from hedonic regression to gradient boosted trees, and the loss choice changes which homes the model gets right [32](https://www.tandfonline.com/doi/full/10.1080/09599916.2022.2070525). Rent and short term rental pricing follow the same recipe [40](https://arxiv.org/pdf/2308.06929); occupancy forecasts lean on bookings already on the books at each lead time (pickup) [54](https://journals.vilniustech.lt/index.php/JBEM/article/view/19775)[55](https://www.sciencedirect.com/science/article/pii/S0278431924001142). Price per square foot is the classic target leak.

#### Property value (automated valuation) (`house_price`)

- **Question**: What is this house or property worth? Segments: small business, large enterprise.
- **Phrasings**: "predict house prices"; "estimate the sale price of each house"; "what is this property worth"; "automated valuation model for homes"; "price homes for listing"; "estimate market value of real estate"; "predict home sale price from features".
- **Framing**: regression; target column like sale_price, saleprice, price, sold_price, close_price, value; one row per property sale.
- **Time**: none
- **Metric**: mae. Home values are right skewed; MAE reads in currency and is not ruled by a few mansions. Split by sale date when the data spans years.
- **Leakage suspects**: `price_per_(sq*|m2|foot|meter)|ppsf|price_sqft|price_psf`, `days_on_market|dom|cumulative_dom`.
- **Strong families**: lightgbm x1.2, catboost x1.2, xgboost x1.1.
- **Columns usually needed**: sqft, bedrooms?, bathrooms?, age*, neighbou?rhood, lot*.
- **Sources**: [32](https://www.tandfonline.com/doi/full/10.1080/09599916.2022.2070525).

#### Rent estimate (`rent_price`)

- **Question**: What monthly rent (or nightly rate) should this unit ask? Segments: small business, large enterprise.
- **Phrasings**: "predict rent for an apartment"; "what rent should we charge for this unit"; "estimate monthly rent from listing features"; "rental price prediction"; "price short term rentals per night"; "predict airbnb nightly price".
- **Framing**: regression; target column like rent, monthly_rent, rent_amount, rent_price, price, nightly_price; one row per unit or listing.
- **Time**: none
- **Metric**: mae. Landlords reason in currency per month; MAE is readable and robust to luxury listings.
- **Leakage suspects**: `rent_per_*|price_per_*|revenue*|booked_nights|occupancy_rate*`.
- **Strong families**: lightgbm x1.2, catboost x1.2.
- **Columns usually needed**: sqft, bedrooms?, bathrooms?, neighbou?rhood, amenit*, room_type.
- **Sources**: [40](https://arxiv.org/pdf/2308.06929).

#### Occupancy forecast (hotel or property) (`occupancy_forecast`)

- **Question**: What share of rooms or units will be occupied each night next month? Segments: small business, large enterprise.
- **Phrasings**: "forecast hotel occupancy"; "predict rooms sold per night"; "forecast occupancy rate for next month"; "how full will the property be next week"; "forecast vacancy for our buildings"; "predict nightly bookings per property".
- **Framing**: regression; target column like occupancy, occupancy_rate, occ, rooms_sold, occupied, occupied_units; one row per property per night or month.
- **Time**: forecast; horizon 1 to 90 nights; the key feature is rooms on the books at the same lead time (pickup); lags 7 and 364 days.
- **Metric**: mae. Revenue managers read the error in rooms or points of occupancy; MAE is readable.
- **Leakage suspects**: `revenue*|revpar|adr_actual|actual_*|no_shows`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, on_the_books, event*, price, property*.
- **Sources**: [54](https://journals.vilniustech.lt/index.php/JBEM/article/view/19775) [55](https://www.sciencedirect.com/science/article/pii/S0278431924001142).

#### Tenant lease renewal (`lease_renewal`)

- **Question**: Will this tenant renew the lease or move out? Segments: small business, large enterprise.
- **Phrasings**: "will the tenant renew their lease"; "predict tenant move outs"; "which residents will not renew"; "lease renewal probability"; "predict tenant turnover for our buildings"; "which commercial tenants will leave at lease end".
- **Framing**: binary; target column like renewed, renewal, will_renew, moved_out, move_out, is_renewed; one row per lease at N days before expiry.
- **Time**: time_split
- **Metric**: roc_auc. Property managers rank leases for retention offers; AUC.
- **Leakage suspects**: `move_out_date|notice_(date|given)|new_lease*|renewal_(rent|date|signed)*|vacated*`.
- **Strong families**: lightgbm x1.2, logreg x1.1.
- **Columns usually needed**: rent*, tenure, maintenance*, late_payments?, unit*.
- **Sources**: none dedicated; completes the catalog.

### Healthcare administration

Prior no shows and booking lead time are the strongest no show predictors [24](https://www.nature.com/articles/s41746-022-00594-w); readmission and length of stay models on EHR data reach AUC around 0.78 with tree models [25](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11401612/); denial models use payer, CPT, ICD and provider specialty before submission [26](https://arxiv.org/pdf/2007.06229)[39](https://www.aapc.com/blog/92229-leveraging-ai-for-denials-management/). DataRobot ships a no show accelerator [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html). Anything recorded at check in, discharge or remittance leaks.

#### Appointment no show (`appointment_no_show`)

- **Question**: Which booked appointments will the patient or client miss? Segments: small business, large enterprise.
- **Phrasings**: "which patients will miss their appointment"; "predict appointment no shows"; "flag appointments at risk of no show for reminders"; "who will not show up to their booking"; "no show prediction for the clinic"; "predict missed visits to overbook slots".
- **Framing**: binary; target column like no_show, noshow, no_show_flag, missed, showed_up, attended; one row per appointment at booking time.
- **Time**: time_split
- **Metric**: roc_auc. Reminder calls and overbooking go to the riskiest slots; AUC. No show rates run 10 to 30 percent.
- **Leakage suspects**: `check_?in*|arrival_(time|at)|arrived*|visit_duration|seen_by|billed*|charge*|cancel*_reason|rescheduled_to*|encounter_id`.
- **Strong families**: lightgbm x1.2, catboost x1.1, logreg x1.1.
- **Columns usually needed**: lead_time*, prior_no_shows?, age, insurance*, sms*, weekday, distance*.
- **Sources**: [24](https://www.nature.com/articles/s41746-022-00594-w) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Hospital readmission within 30 days (`readmission`)

- **Question**: Will this patient be readmitted within 30 days of discharge? Segments: large enterprise.
- **Phrasings**: "predict 30 day readmission"; "which patients will be readmitted"; "readmission risk at discharge"; "flag patients likely to come back to hospital"; "predict unplanned readmissions"; "hospital readmission model for care management".
- **Framing**: binary; target column like readmitted, readmit, readmission, readmit_30, readmitted_30, is_readmitted; one row per discharge (encounter).
- **Time**: time_split
- **Metric**: roc_auc. Care managers call the riskiest discharges; AUC is the standard metric in readmission studies.
- **Leakage suspects**: `readmi*_(date|id|days)|days_to_readmi*|next_admission*|post_discharge*|followup_(visit|outcome)*`.
- **Strong families**: lightgbm x1.2, logreg x1.2, catboost x1.1.
- **Columns usually needed**: age, diag*, length_of_stay, num_*, medications?, discharge_disposition*, lab*.
- **Sources**: [25](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11401612/) [2](https://h2o.ai/solutions/use-case/) [1](https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html).

#### Length of stay (`length_of_stay`)

- **Question**: How many days will this patient stay in the hospital? Segments: large enterprise.
- **Phrasings**: "predict length of stay"; "how many days will the patient stay"; "estimate hospital stay duration at admission"; "forecast bed days per admission"; "predict los for inpatients"; "which admissions will have a long stay".
- **Framing**: regression; target column like los, length_of_stay, lengthofstay, los_days, stay_days, time_in_hospital; one row per admission.
- **Time**: time_split
- **Metric**: mae. Stays are right skewed; MAE in days is what bed managers plan with.
- **Leakage suspects**: `discharge_(date|time|disposition|status)*|discharged*|total_charges|total_costs|billed*`.
- **Strong families**: lightgbm x1.2, catboost x1.2.
- **Columns usually needed**: admission_type*, diag*, age*, comorbid*, admission_source*, weekday.
- **Sources**: [25](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11401612/).

#### Medical claim denial (`claim_denial`)

- **Question**: Will the payer deny this claim, so we can fix it before submission? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict which claims will be denied"; "claim denial prediction before submission"; "flag medical claims likely to be rejected by the payer"; "which insurance claims will bounce"; "reduce denials in revenue cycle"; "predict payer rejections".
- **Framing**: binary; target column like denied, is_denied, denial, rejected, is_rejected, claim_status; one row per claim (or claim line) before submission.
- **Time**: time_split
- **Metric**: roc_auc. Billers review the riskiest claims first; AUC measures that ranking.
- **Leakage suspects**: `denial_(code|reason|date)|carc*|rarc*|remit*|adjustment_(reason|code)|paid_amount|payment_(date|amount)|allowed_amount|appeal*`.
- **Strong families**: catboost x1.3, lightgbm x1.1.
- **Columns usually needed**: payer*, cpt*, icd*, specialty, place_of_service, prior_auth*, billed_amount.
- **Sources**: [26](https://arxiv.org/pdf/2007.06229) [39](https://www.aapc.com/blog/92229-leveraging-ai-for-denials-management/) [2](https://h2o.ai/solutions/use-case/).

#### Patient volume forecast (`patient_volume_forecast`)

- **Question**: How many patients will arrive per day or hour at each clinic or ward? Segments: small business, large enterprise.
- **Phrasings**: "forecast patient visits per day"; "predict emergency department arrivals"; "how many patients will we see next week"; "forecast admissions for bed planning"; "predict clinic volume for staffing"; "forecast daily hospital census".
- **Framing**: regression; target column like visits, arrivals, patients, admissions, census, volume; one row per clinic or ward per day (or hour).
- **Time**: forecast; horizon 1 to 14 days; lags 7, 14 and 364 days; flu season, holidays and the weather forecast.
- **Metric**: mae. Staffing plans read the error as patients per shift; MAE is readable.
- **Leakage suspects**: `discharges_same*|lwbs|left_without*|actual_*`.
- **Strong families**: lightgbm x1.4, xgboost x1.1, catboost x1.1, random_forest x0.8, mlp x0.7, tabicl x0.6.
- **Columns usually needed**: date, day_of_week, holiday*, flu*, temp*, clinic*.
- **Sources**: [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [2](https://h2o.ai/solutions/use-case/).

### Manufacturing

In AI4I 2020 the five failure mode columns (TWF, HDF, PWF, OSF, RNF) jointly define Machine failure, so keeping them leaks the label [19](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Azure's predictive maintenance accelerator predicts remaining useful life of turbofan engines [4](https://msusazureaccelerators.github.io/accelerators/demand-forecasting.html); AWS has fleet failure templates [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html)[37](https://aws.amazon.com/blogs/machine-learning/predict-vehicle-fleet-failure-probability-using-amazon-sagemaker-jumpstart/). Quality and yield models use process settings and sensor summaries [43](https://dac.digital/machine-learning-use-cases-in-manufacturing/); warranty models link claims to build batches and suppliers [53](https://www.researchgate.net/publication/335503034_Vehicle_Warranty_Claim_Prediction_from_Diagnostic_Data_Using_Classification).

#### Predictive maintenance (failure in the next N days) (`predictive_maintenance`)

- **Question**: Which machines will fail in the next N days? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict machine failures"; "which machines will break down next week"; "predictive maintenance model from sensor data"; "flag equipment likely to fail soon"; "predict failures in the next 30 days"; "detect assets at risk of breakdown"; "when should we service each machine".
- **Framing**: binary; target column like failure, machine_failure, failed, will_fail, fail, breakdown; one row per machine per day (or cycle).
- **Time**: time_split
- **Metric**: f1_macro. Failures are rare and a missed one is costly; f1_macro scores the maintenance flag instead of a flattering AUC.
- **Leakage suspects**: `twf|hdf|pwf|osf|rnf`, `failure_(type|mode|code|date)|repair*|downtime*|maintenance_ticket*|replaced*|work_order*`.
- **Strong families**: lightgbm x1.2, xgboost x1.2, random_forest x1.1.
- **Columns usually needed**: temp*, vibration*, pressure*, rpm, torque*, tool_wear*, machine*.
- **Sources**: [19](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html) [37](https://aws.amazon.com/blogs/machine-learning/predict-vehicle-fleet-failure-probability-using-amazon-sagemaker-jumpstart/) [2](https://h2o.ai/solutions/use-case/).

#### Remaining useful life (`remaining_useful_life`)

- **Question**: How many cycles or days does this asset have left before it fails? Segments: large enterprise.
- **Phrasings**: "predict remaining useful life"; "how many cycles until the engine fails"; "estimate time to failure for each asset"; "rul prediction from sensor readings"; "predict days until the next breakdown"; "how long before this component wears out".
- **Framing**: regression; target column like rul, remaining_useful_life, cycles_to_failure, time_to_failure, ttf, days_to_failure; one row per asset per cycle.
- **Time**: time_split
- **Metric**: rmse. Large misses near end of life are the dangerous ones; RMSE (the NASA turbofan benchmark metric) punishes them.
- **Leakage suspects**: `max_cycles?|total_cycles|end_of_life|failure_(cycle|date)|last_cycle`.
- **Strong families**: lightgbm x1.2, xgboost x1.1, random_forest x0.9.
- **Columns usually needed**: unit*, cycle*, sensor*, setting*.
- **Sources**: [4](https://msusazureaccelerators.github.io/accelerators/demand-forecasting.html) [3](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html).

#### Quality defect prediction (`quality_defect`)

- **Question**: Will this part, batch or product fail quality inspection? Segments: small business, mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict defective parts"; "flag products likely to fail quality inspection"; "which batches will be scrapped"; "predict pass or fail from process data"; "quality prediction from machine settings"; "detect defects before final inspection".
- **Framing**: binary; target column like defect, defective, is_defective, pass_fail, pass, fail; one row per part or batch.
- **Time**: time_split
- **Metric**: f1_macro. Defects are rare and the line acts on a pass or fail flag; f1_macro scores that flag.
- **Leakage suspects**: `defect_(type|code|reason|count)|scrap_(reason|qty|cost)|rework*|inspection_(result|outcome|notes?)|qc_*|final_test*|returned*`.
- **Strong families**: lightgbm x1.2, xgboost x1.2, random_forest x1.1.
- **Columns usually needed**: temp*, pressure*, speed*, machine*, material*, shift, duration*.
- **Sources**: [43](https://dac.digital/machine-learning-use-cases-in-manufacturing/).

#### Process yield (`yield_prediction`)

- **Question**: What yield or output will this batch or production run achieve? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict process yield per batch"; "forecast production output for each run"; "what yield will we get from these settings"; "predict throughput of the line"; "estimate batch yield from raw material properties"; "predict crop or plant yield per field".
- **Framing**: regression; target column like yield, yield_pct, yield_percent, yield_rate, output, throughput; one row per batch or production run.
- **Time**: time_split
- **Metric**: rmse. Yield sits in a narrow band and large shortfalls are what hurt; RMSE punishes them. Use mae if a few bad batches are data errors.
- **Leakage suspects**: `good_units|scrap_(units|qty)|reject_(units|qty)|final_weight|output_weight|defects?_count`.
- **Strong families**: lightgbm x1.2, xgboost x1.1.
- **Columns usually needed**: temp*, pressure*, duration*, material*, equipment*, shift.
- **Sources**: [43](https://dac.digital/machine-learning-use-cases-in-manufacturing/).

#### Fault or defect type classification (`defect_type`)

- **Question**: Which kind of fault or defect is this? Segments: large enterprise.
- **Phrasings**: "classify the type of machine fault"; "which failure mode caused the stop"; "predict defect category for each part"; "diagnose the fault type from sensor readings"; "classify steel plate faults"; "root cause category for each defect".
- **Framing**: multiclass; target column like defect_type, fault_type, failure_type, failure_mode, fault, fault_class; one row per defect event or part.
- **Time**: none
- **Metric**: f1_macro. Some fault types are rare but matter as much as common ones; f1_macro weighs each class equally.
- **Leakage suspects**: `repair_action*|corrective_action*|resolution*|technician_notes?`.
- **Strong families**: lightgbm x1.2, xgboost x1.1, random_forest x1.1.
- **Columns usually needed**: sensor*, temp*, machine*, x_*.
- **Sources**: [48](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) [19](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

#### Warranty claim prediction (`warranty_claim`)

- **Question**: Which sold units will come back with a warranty claim? Segments: mid-market B2B SaaS, large enterprise.
- **Phrasings**: "predict warranty claims"; "which products will have a warranty claim"; "forecast warranty failures by build batch"; "flag units likely to fail under warranty"; "predict returns under warranty for sold devices"; "early warning for warranty issues".
- **Framing**: binary; target column like warranty_claim, claimed, has_claim, claim, failed_in_warranty, returned; one row per unit sold.
- **Time**: time_split
- **Metric**: roc_auc. Quality teams rank build batches and units for inspection; AUC.
- **Leakage suspects**: `claim_(date|amount|cost|code)|repair_(cost|date|code)*|failure_code*|replacement*|rma*`.
- **Strong families**: lightgbm x1.2, catboost x1.1.
- **Columns usually needed**: model*, plant*, build_date, supplier*, usage*, region.
- **Sources**: [53](https://www.researchgate.net/publication/335503034_Vehicle_Warranty_Claim_Prediction_from_Diagnostic_Data_Using_Classification).

## Out of scope for the factory

Goals that sound tabular but need a different tool; the matcher scores them low or maps them to the nearest supervised framing, and the agent should say so.

- Uplift or treatment effect ("who should get the coupon so that they buy"): needs a causal design and control groups, not a plain classifier.
- Recommendation ranking over thousands of items: collaborative filtering, not one row per customer. `next_best_offer` covers the few offers case.
- Segmentation or clustering with no target, and anomaly detection with no labels.
- Free text, images or audio as the main signal (tickets, call transcripts, photos); tabular summaries of them are fine.
- Optimization (price setting, scheduling, routing): the factory supplies the prediction an optimizer needs.

## Sources

1. DataRobot, use cases and horizontal approaches (accelerator library): https://docs.datarobot.com/11.0/en/docs/api/accelerators/use-cases-accel/index.html
2. H2O.ai, use case library by industry: https://h2o.ai/solutions/use-case/
3. AWS, SageMaker JumpStart end to end solution templates: https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-solutions.html
4. Microsoft, Azure accelerators: demand forecasting and predictive maintenance: https://msusazureaccelerators.github.io/accelerators/demand-forecasting.html
5. Microsoft, Azure ML many models solution accelerator: https://learn.microsoft.com/en-us/samples/azure/machinelearningnotebooks/many-models-solution-accelerator/
6. Google Cloud, Vertex AI introduction to tabular data: https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular101
7. Google, crystalvalue predictive LTV on Vertex AI: https://github.com/google-marketing-solutions/crystalvalue
8. McKinsey, reducing churn in telecom through advanced analytics: https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/reducing-churn-in-telecom-through-advanced-analytics
9. McKinsey, B2B pricing: navigating the next phase of the AI revolution: https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution
10. Gartner, AI use case prism for supply chain: https://www.gartner.com/en/documents/3999399
11. Makridakis et al., M5 accuracy competition: results, findings and conclusions: https://www.sciencedirect.com/science/article/pii/S0169207021001874
12. Hyndman and Athanasopoulos, Forecasting: Principles and Practice, evaluating point forecast accuracy: https://otexts.com/fpp3/accuracy.html
13. Kaggle, Home Credit default risk: https://www.kaggle.com/competitions/home-credit-default-risk
14. Kaggle, Allstate claims severity (evaluation: MAE): https://www.kaggle.com/competitions/allstate-claims-severity/overview/evaluation
15. Kaggle, Porto Seguro safe driver prediction (normalized Gini): https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
16. Kaggle, Rossmann store sales: https://www.kaggle.com/c/rossmann-store-sales
17. Kaggle, playground S3E4 credit card fraud: https://www.kaggle.com/competitions/playground-series-s3e4
18. UCI, Bank Marketing dataset (duration must be discarded): https://archive.ics.uci.edu/dataset/222/bank+marketing
19. UCI, AI4I 2020 predictive maintenance dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
20. Leakage audited Lending Club scorecard (naive AUC vs origination only): https://github.com/alyrraza/pit-scorecard
21. Hotel booking cancellation project (reservation_status leak): https://github.com/manarsabryqassem32-web/End-to-end-Hotel-Booking-Cancellation-Prediction
22. Dataconomy, label leakage in churn models: https://dataconomy.com/2026/09/22/label-leakage-the-bug-that-makes-churn-models-look-brilliant/
23. Leakage free evaluation for employee attrition prediction (Information, 2026): https://www.mdpi.com/2078-2489/17/3/308
24. npj Digital Medicine, machine learning for pediatric appointment no shows: https://www.nature.com/articles/s41746-022-00594-w
25. In hospital mortality, readmission and prolonged LOS prediction from EHR: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11401612/
26. Deep Claim: payer response prediction from claims data: https://arxiv.org/pdf/2007.06229
27. Optimize cash collection: predicting invoice payment with ML: https://arxiv.org/pdf/1912.10828
28. Predicting late supplier deliveries in low volume high variety machinery: https://www.sciencedirect.com/science/article/pii/S294986352300002X
29. Salesforce Ben, Einstein opportunity scoring: https://www.salesforceben.com/what-is-salesforce-einstein-opportunity-scoring/
30. ProductLed, guide to product qualified leads: https://productled.com/blog/product-qualified-leads
31. Wang et al., a deep probabilistic model for customer lifetime value (ZILN): https://arxiv.org/pdf/1912.07753
32. House price prediction with gradient boosted trees under different loss functions: https://www.tandfonline.com/doi/full/10.1080/09599916.2022.2070525
33. Capsule CRM, small business AI adoption statistics: https://capsulecrm.com/blog/small-business-ai-adoption-statistics/
34. PayPal, small business AI survey (June 2025): https://newsroom.paypal-corp.com/2025-06-10-Beyond-Efficiency-Small-Businesses-Look-to-AI-for-Competitive-Edge,-New-Survey-Shows
35. Accoil, customer health score guide for B2B SaaS: https://www.accoil.com/blog/customer-health-score
36. Wipro, machine learning for B2B pricing: https://www.wipro.com/analytics/machine-learning-for-b2b-pricing/
37. AWS, predict vehicle fleet failure probability: https://aws.amazon.com/blogs/machine-learning/predict-vehicle-fleet-failure-probability-using-amazon-sagemaker-jumpstart/
38. Predicting account receivables with machine learning: https://arxiv.org/html/2008.07363v1
39. AAPC, leveraging AI for denials management: https://www.aapc.com/blog/92229-leveraging-ai-for-denials-management/
40. Predicting listing prices in short term rental markets: https://arxiv.org/pdf/2308.06929
41. Kaggle, IBM HR analytics attrition dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
42. AI to predict supplier lead times in procurement: https://zenodo.org/records/17838266
43. DAC.digital, machine learning use cases in manufacturing: https://dac.digital/machine-learning-use-cases-in-manufacturing/
44. Hyndman and Koehler, another look at measures of forecast accuracy: https://robjhyndman.com/papers/mase.pdf
45. Kaggle, playground S4E7 insurance cross selling: https://www.kaggle.com/competitions/playground-series-s4e7
46. UCI, online shoppers purchasing intention dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
47. Kaggle, fraud detection in insurance claims: https://www.kaggle.com/competitions/fraud-detection-in-insurance-claims
48. UCI, steel plates faults dataset: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
49. Returns foresight: explainable ML for product return propensity in apparel: https://www.sciencedirect.com/science/article/abs/pii/S0969698926001086
50. Garment returns prediction: comparison of ML algorithms: https://link.springer.com/article/10.1007/s42979-025-03944-z
51. Job acceptance prediction system (HR analytics): https://github.com/kousiksiva/Job-Acceptance-Prediction-System
52. Noon, predictive hiring analytics guide: https://www.noon.ai/blog/articles/21-predictive-hiring-analytics-guide
53. Vehicle warranty claim prediction from diagnostic data: https://www.researchgate.net/publication/335503034_Vehicle_Warranty_Claim_Prediction_from_Diagnostic_Data_Using_Classification
54. Application of machine learning algorithms to predict hotel occupancy: https://journals.vilniustech.lt/index.php/JBEM/article/view/19775
55. Interpretable ML for hotel occupancy forecasting with pickup: https://www.sciencedirect.com/science/article/pii/S0278431924001142
