import copy


def compute_average_review(reviews, uncertainty_mode='keep'):
    """

    Args:
        reviews: List of dictionaries containing reviews
        uncertainty_mode: Accepts `drop` (discard uncertainty), `keep`(keep uncertainty value in the
            review dictionary) and `distribute` (distribute the uncertainty between other diagnosis) as values
    Returns: Returns the normalized final review. A dictionary containing {condition: probability} for all the
    conditions in the review

    """
    if uncertainty_mode not in ['distribute', 'drop', 'keep']:
        raise ValueError('uncertainty_mode not allowed. We only allow `keep`, `distribute` or `drop`.')

    reviews = copy.deepcopy(reviews)
    average_review = {}
    n_reviewers = len(reviews)
    for review in reviews:
        uncertainty_to_distribute = 0
        if 'uncertainty' in review and uncertainty_mode != 'keep':
            if len(review) == 1:
                n_reviewers -= 1
            elif uncertainty_mode == 'distribute':
                uncertainty_to_distribute = review['uncertainty'] / (len(review) - 1)
            del review['uncertainty']
        for diagnosis, confidence in review.items():
            if diagnosis in average_review.keys():
                average_review[diagnosis] = average_review[diagnosis] + confidence + uncertainty_to_distribute
            else:
                average_review[diagnosis] = confidence + uncertainty_to_distribute
    divisor = n_reviewers * 100
    if uncertainty_mode == 'drop':
        divisor = sum(average_review.values())
    for key, value in average_review.items():
        average_review[key] = value / divisor
    return average_review


def keep_outliers(diagnoses, model_map):
    # Merges DN children to DN, keeps everything else
    mapped_diagnoses = {}
    reviews_out = {}
    for key, value in diagnoses.items():
        if key in model_map.conditions2diagnosis_map:
            diagnosis = model_map.conditions2diagnosis_map[key]
            mapped_diagnoses[diagnosis] = mapped_diagnoses.get(diagnosis, 0) + value
        else:
            reviews_out.update({key: value})
    if len(reviews_out) > 0:
        mapped_diagnoses.update(reviews_out)

    return mapped_diagnoses


def reject_outliers(diagnoses, model_map):
    # Merges DN children to DN, reject reviews with outside nodes
    mapped_diagnoses = {}
    for key, value in diagnoses.items():
        if key in model_map.conditions2diagnosis_map:
            diagnosis = model_map.conditions2diagnosis_map[key]
            mapped_diagnoses[diagnosis] = mapped_diagnoses.get(diagnosis, 0) + value
        else:
            if key in ('uncertainty', 'bad_quality'):
                mapped_diagnoses.update({key: value})
            else:
                # End loop and reject review
                mapped_diagnoses = {}
                break

    return mapped_diagnoses


def outliers_as_uncertainty(diagnoses, model_map):
    # Merges DN children to DN, adds outlier probabilities to uncertainty class
    mapped_diagnoses = {}
    for key, value in diagnoses.items():
        if key in model_map.conditions2diagnosis_map:
            diagnosis = model_map.conditions2diagnosis_map[key]
            mapped_diagnoses[diagnosis] = mapped_diagnoses.get(diagnosis, 0) + value
        else:
            if key == 'bad_quality':
                mapped_diagnoses.update({key: value})
            else:
                mapped_diagnoses['uncertainty'] = mapped_diagnoses.get('uncertainty', 0) + value

    return mapped_diagnoses


# Closure function to pass ancestor_set and ontology
def distribute_ancestors_to_dn_outliers_to_uncertainty(ancestors_diagnosis_ids_map):
    # Merges DN children to DN, distributes ancestor probabilities to their children DN, adds outliers probabilities to
    # uncertainty class
    def mapping(diagnoses, model_map):
        mapped_diagnoses = {}
        for key, value in diagnoses.items():
            if key in model_map.conditions2diagnosis_map:
                diagnosis = model_map.conditions2diagnosis_map[key]
                mapped_diagnoses[diagnosis] = mapped_diagnoses.get(diagnosis, 0) + value
            else:
                if key in ancestors_diagnosis_ids_map:
                    diagnosis_nodes = ancestors_diagnosis_ids_map[key]
                    probability = value / len(diagnosis_nodes)
                    for diagnosis in diagnosis_nodes:
                        mapped_diagnoses[diagnosis] = mapped_diagnoses.get(diagnosis, 0) + probability
                else:
                    mapped_diagnoses['uncertainty'] = mapped_diagnoses.get('uncertainty', 0) + value

        return mapped_diagnoses
    return mapping


def uncertainty_is_above_threshold(reviews, threshold=0.5):
    """

    Args:
        reviews: List of dictionaries containing reviews
        threshold: Threshold parameter to compare uncertainty with, range has to be between [0, 1]

    Returns: True if mean uncertainty is above threshold

    """
    uncertainty = [review['uncertainty'] for review in reviews if 'uncertainty' in review]
    if len(uncertainty) > 0:
        return sum(uncertainty) / (100 * len(reviews)) >= threshold
    return False
