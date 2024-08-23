import jax
import jax.numpy as jnp
import optax

from networks.resnet import ResNet18


def create_train_state(rng, learning_rate, input_shape):
    global_client_net, server_net = ResNet18()
    params = global_client_net.init(rng, jnp.zeros(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)



@jax.jit
def train_step(state, batch):
    x, y = batch
    loss_grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = loss_grad_fn(state.params, x, y, state.apply_fn)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(state, train_ds, num_epochs, batch_size):
    for epoch in range(num_epochs):
        rng = jax.random.PRNGKey(epoch)

        train_ds = jax.random.permutation(rng, train_ds)

        batches = jnp.array_split(train_ds, len(train_ds) // batch_size)

        epoch_loss = 0.0
        for batch in batches:
            state, loss = train_step(state, batch)
            epoch_loss += loss

        avg_loss = epoch_loss / len(batches)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return state


if __name__ == "__main__":
    client_net, server_net = ResNet18()
    print(client_net)
