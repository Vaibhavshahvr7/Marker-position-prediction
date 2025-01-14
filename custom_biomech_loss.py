import tensorflow as tf

def pred_restric_loss(y_true, y_pred):
    """Calculates a loss based on the differences between predictions and true values
    at progressively larger temporal intervals. This ensures the model's predictions
    maintain consistency over time.

    Args:
        y_true (tensor): Ground truth values with shape [batch, time, height, width].
        y_pred (tensor): Predicted values with shape [batch, time, height, width].

    Returns:
        tensor: The restricted prediction loss.
    """
    # Compute differences between adjacent time steps at various intervals for predictions
    yp_1 = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]  # 1-time-step difference
    yp_2 = y_pred[:,2:,:,:] - y_pred[:,:-2,:,:]  # 2-time-step difference
    yp_3 = y_pred[:,3:,:,:] - y_pred[:,:-3,:,:]  # 3-time-step difference
    yp_4 = y_pred[:,4:,:,:] - y_pred[:,:-4,:,:]  # 4-time-step difference
    yp_5 = y_pred[:,5:,:,:] - y_pred[:,:-5,:,:]  # 5-time-step difference

    # Compute differences between adjacent time steps at various intervals for ground truth
    yt_1 = y_true[:,1:,:,:] - y_true[:,:-1,:,:]
    yt_2 = y_true[:,2:,:,:] - y_true[:,:-2,:,:]
    yt_3 = y_true[:,3:,:,:] - y_true[:,:-3,:,:]
    yt_4 = y_true[:,4:,:,:] - y_true[:,:-4,:,:]
    yt_5 = y_true[:,5:,:,:] - y_true[:,:-5,:,:]

    # Calculate mean squared error (MSE) for each temporal interval difference
    ym1 = tf.reduce_mean(tf.square(yp_1 - yt_1))
    ym2 = tf.reduce_mean(tf.square(yp_2 - yt_2))
    ym3 = tf.reduce_mean(tf.square(yp_3 - yt_3))
    ym4 = tf.reduce_mean(tf.square(yp_4 - yt_4))
    ym5 = tf.reduce_mean(tf.square(yp_5 - yt_5))

    # Aggregate MSEs for final loss value
    mse1 = ym1 + ym2 + ym3 + ym4 + ym5

    # Scale the loss by a factor of 2
    mse = mse1 * 2
    return mse

def mse_loss(y_true, y_pred):
    """Calculates the Mean Squared Error (MSE) loss.

    Args:
        y_true (tensor): Ground truth values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: The MSE loss value.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mse_loss_top_24(y_true, y_pred):
    """Calculates the MSE for the top 24 most significant error values.

    Args:
        y_true (tensor): Ground truth values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: The averaged MSE of the top 24 largest errors.
    """
    # Compute the mean squared error along the batch and temporal axes
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
    mse = tf.reduce_mean(mse, axis=0)

    # Reshape and sort the MSE values in descending order
    mse = tf.reshape(mse, [48])
    mse = tf.sort(mse, direction='DESCENDING')

    # Select the top 24 largest errors and compute their mean
    mse = mse[:24]
    return tf.reduce_mean(mse)

@tf.function
def custom_biomech_loss(y_true, y_pred):
    """Custom biomechanical loss function to account for both spatial and temporal constraints.

    Args:
        y_true (tensor): Ground truth values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: The computed loss value.
    """
    # Compute the overall MSE
    mse = mse_loss(y_true, y_pred)

    # Calculate separate MSEs for specific biomechanical features (e.g., right and left foot)
    mse_R_foot = mse_loss(y_true[..., 7:9], y_pred[..., 7:9])
    mse_L_foot = mse_loss(y_true[..., 13:15], y_pred[..., 13:15])

    # Add the feature-specific losses to the overall MSE
    mse = mse + mse_R_foot + mse_L_foot

    # Add the prediction restriction loss and top-24 MSE to the total loss
    mse_1 = pred_restric_loss(y_true, y_pred)
    mse_2 = mse_loss_top_24(y_true, y_pred)
    mse = 2 * mse + 4 * mse_1 + mse_2

    # Take the square root of the final loss to scale it down
    mse = tf.sqrt(mse)
    return mse
