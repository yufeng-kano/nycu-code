<script setup lang="ts">
import { onMounted, ref } from "vue";
import { addName, getNames, removeName, type NameRecord } from "./api";

const names = ref<NameRecord[]>([]);
const input = ref("");
const error = ref("");
const busy = ref(false);

async function refresh(): Promise<void> {
  error.value = "";
  try {
    names.value = (await getNames()).names;
  } catch (e) {
    error.value = (e as Error).message;
  }
}

async function submit(): Promise<void> {
  const name = input.value.trim();
  if (!name || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    await addName(name);
    input.value = "";
    await refresh();
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    busy.value = false;
  }
}

async function remove(name: string): Promise<void> {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    await removeName(name);
    await refresh();
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    busy.value = false;
  }
}

onMounted(refresh);
</script>

<template>
  <main class="container">
    <h1>Names</h1>

    <form class="row" @submit.prevent="submit">
      <input
        v-model="input"
        type="text"
        placeholder="Enter a name"
        autocomplete="off"
        :disabled="busy"
      />
      <button type="submit" :disabled="busy || !input.trim()">Add</button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>

    <p v-if="names.length === 0" class="empty">No names yet.</p>
    <ul v-else>
      <li v-for="item in names" :key="item.id" class="row">
        <span>{{ item.name }}</span>
        <button type="button" class="danger" :disabled="busy" @click="remove(item.name)">
          Remove
        </button>
      </li>
    </ul>
  </main>
</template>
